/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

/* Algorithm: Rabenseifner's Algorithm
 *
 * Restrictions: Built-in ops only
 *
 * This algorithm is from http://www.hlrs.de/mpi/myreduce.html.
.
 * This algorithm implements the allreduce in two steps: first a
 * reduce-scatter, followed by an allgather. A recursive-halving algorithm
 * (beginning with processes that are distance 1 apart) is used for the
 * reduce-scatter, and a recursive doubling algorithm is used for the
 * allgather. The non-power-of-two case is handled by dropping to the nearest
 * lower power-of-two: the first few even-numbered processes send their data to
 * their right neighbors (rank+1), and the reduce-scatter and allgather happen
 * among the remaining power-of-two processes. At the end, the first few
 * even-numbered processes get the result from their right neighbors.
 *
 * For the power-of-two case, the cost for the reduce-scatter is:
 *
 * lgp.alpha + n.((p-1)/p).beta + n.((p-1)/p).gamma.
 *
 * The cost for the allgather:
 *
 * lgp.alpha +.n.((p-1)/p).beta
 *
 * Therefore, the total cost is:
 *
 * Cost = 2.lgp.alpha + 2.n.((p-1)/p).beta + n.((p-1)/p).gamma
 *
 * For the non-power-of-two case:
 *
 * Cost = (2.floor(lgp)+2).alpha + (2.((p-1)/p) + 2).n.beta + n.(1+(p-1)/p).gamma
 */

int MPIR_Allreduce_intra_reduce_scatter_allgather(const void *sendbuf,
                                                  void *recvbuf,
                                                  MPI_Aint count,
                                                  MPI_Datatype datatype,
                                                  MPI_Op op, MPIR_Comm * comm_ptr, int coll_attr)
{
    MPIR_CHKLMEM_DECL();
    int comm_size, rank;
    int mpi_errno = MPI_SUCCESS;
    int mask, dst, pof2, newrank, rem, newdst, i, send_idx, recv_idx, last_idx;
    MPI_Aint true_extent, true_lb, extent;
    void *tmp_buf;

    MPIR_COMM_RANK_SIZE(comm_ptr, rank, comm_size);

    /* need to allocate temporary buffer to store incoming data */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPIR_Datatype_get_extent_macro(datatype, extent);

    MPIR_CHKLMEM_MALLOC(tmp_buf, count * (MPL_MAX(extent, true_extent)));

    /* adjust for potential negative lower bound in datatype */
    tmp_buf = (void *) ((char *) tmp_buf - true_lb);

    /* copy local data into recvbuf */
    if (sendbuf != MPI_IN_PLACE) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf, count, datatype);
        MPIR_ERR_CHECK(mpi_errno);
    }

    /* get nearest power-of-two less than or equal to comm_size */
    pof2 = MPL_pof2(comm_size);

    rem = comm_size - pof2;

    /* In the non-power-of-two case, all even-numbered
     * processes of rank < 2*rem send their data to
     * (rank+1). These even-numbered processes no longer
     * participate in the algorithm until the very end. The
     * remaining processes form a nice power-of-two. */

    if (rank < 2 * rem) {
        if (rank % 2 == 0) {    /* even */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank + 1, MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            /* temporarily set the rank to -1 so that this
             * process does not pariticipate in recursive
             * doubling */
            newrank = -1;
        } else {        /* odd */
            mpi_errno = MPIC_Recv(tmp_buf, count,
                                  datatype, rank - 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);

            /* do the reduction on received data. since the
             * ordering is right, it doesn't matter whether
             * the operation is commutative or not. */
            mpi_errno = MPIR_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* change the rank */
            newrank = rank / 2;
        }
    } else      /* rank >= 2*rem */
        newrank = rank - rem;

    /* If op is user-defined or count is less than pof2, use
     * recursive doubling algorithm. Otherwise do a reduce-scatter
     * followed by allgather. (If op is user-defined,
     * derived datatypes are allowed and the user could pass basic
     * datatypes on one process and derived on another as long as
     * the type maps are the same. Breaking up derived
     * datatypes to do the reduce-scatter is tricky, therefore
     * using recursive doubling in that case.) */

#ifdef HAVE_ERROR_CHECKING
    MPIR_Assert(HANDLE_IS_BUILTIN(op));
    MPIR_Assert(count >= pof2);
#endif /* HAVE_ERROR_CHECKING */

    if (newrank != -1) {
        MPI_Aint *cnts, *disps;
        MPIR_CHKLMEM_MALLOC(cnts, pof2 * sizeof(MPI_Aint));
        MPIR_CHKLMEM_MALLOC(disps, pof2 * sizeof(MPI_Aint));

        for (i = 0; i < pof2; i++)
            cnts[i] = count / pof2;
        if ((count % pof2) > 0) {
            for (i = 0; i < (count % pof2); i++)
                cnts[i] += 1;
        }

        if (pof2)
            disps[0] = 0;
        for (i = 1; i < pof2; i++)
            disps[i] = disps[i - 1] + cnts[i - 1];

        mask = 0x1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask < pof2) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            }

            /* Send data from recvbuf. Recv into tmp_buf */
            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) tmp_buf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            /* tmp_buf contains data received in this step.
             * recvbuf contains data accumulated so far */

            /* This algorithm is used only for predefined ops
             * and predefined ops are always commutative. */
            mpi_errno = MPIR_Reduce_local(((char *) tmp_buf + disps[recv_idx] * extent),
                                          ((char *) recvbuf + disps[recv_idx] * extent),
                                          recv_cnt, datatype, op);
            MPIR_ERR_CHECK(mpi_errno);

            /* update send_idx for next iteration */
            send_idx = recv_idx;
            mask <<= 1;

            /* update last_idx, but not in last iteration
             * because the value is needed in the allgather
             * step below. */
            if (mask < pof2)
                last_idx = recv_idx + pof2 / mask;
        }

        /* now do the allgather */

        mask >>= 1;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            MPI_Aint send_cnt, recv_cnt;
            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pof2 / 2)
                    last_idx = last_idx + pof2 / (mask * 2);

                recv_idx = send_idx + pof2 / (mask * 2);
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += cnts[i];
            } else {
                recv_idx = send_idx - pof2 / (mask * 2);
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += cnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += cnts[i];
            }

            mpi_errno = MPIC_Sendrecv((char *) recvbuf +
                                      disps[send_idx] * extent,
                                      send_cnt, datatype,
                                      dst, MPIR_ALLREDUCE_TAG,
                                      (char *) recvbuf +
                                      disps[recv_idx] * extent,
                                      recv_cnt, datatype, dst,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE, coll_attr);
            MPIR_ERR_CHECK(mpi_errno);

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
        }
    }
    /* In the non-power-of-two case, all odd-numbered
     * processes of rank < 2*rem send the result to
     * (rank-1), the ranks who didn't participate above. */
    if (rank < 2 * rem) {
        if (rank % 2)   /* odd */
            mpi_errno = MPIC_Send(recvbuf, count,
                                  datatype, rank - 1, MPIR_ALLREDUCE_TAG, comm_ptr, coll_attr);
        else    /* even */
            mpi_errno = MPIC_Recv(recvbuf, count,
                                  datatype, rank + 1,
                                  MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE);
        MPIR_ERR_CHECK(mpi_errno);
    }
  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
