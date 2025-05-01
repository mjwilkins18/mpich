/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

#ifdef ENABLE_ONECCL

#include <sycl/sycl.hpp>
#include <ccl.hpp>
#include <exception>
#include <string>

/*
 * Exception class for OneCCL errors
 */
class OneCCLException : public std::exception {
private:
    std::string message;

public:
    explicit OneCCLException(const std::string& msg) : message(msg) {}
    virtual const char* what() const noexcept override {
        return message.c_str();
    }
};

/*
 * OneCCL communicator class 
 */
class MPIR_OneCCLcomm {
public:
    MPIR_OneCCLcomm(int rank) {
        sycl::device device;
        sycl::queue queue;
        auto platform_list = sycl::platform::get_platforms();
        bool device_found = false;

        for (const auto &platform : platform_list) {
            auto platform_name = platform.get_info<sycl::info::platform::name>();
            bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
            if (is_level_zero) {
                auto device_list = platform.get_devices();
                for (const auto &dev : device_list) {
                    if (dev.is_gpu()) {
                        device = dev;
                        sycl::context context(device);
                        queue = sycl::queue(context, device, {sycl::property::queue::in_order()});
                        device_found = true;
                        break;
                    }
                }
            }
            if (device_found) break;
        }

        if (!device_found) {
            throw OneCCLException("No Level-Zero GPU device found for rank " + std::to_string(rank));
        }

        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type main_addr;
        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            if (MPIR_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
                throw OneCCLException("MPI_Bcast failed");
            }
        } else {
            if (MPIR_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
                throw OneCCLException("MPI_Bcast failed");
            }
            kvs = ccl::create_kvs(main_addr);
        }

        auto dev = ccl::create_device(queue.get_device());
        auto ctx = ccl::create_context(queue.get_context());
        comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
        stream = ccl::create_stream(queue);
    }

    ~MPIR_OneCCLcomm() = default;

    ccl::event allreduce(const void *sendbuf, void *recvbuf, size_t count, ccl:datatype datatype, ccl::reduction op)
    {
        return ccl::allreduce(sendbuf, recvbuf, count, op, datatype, comm, stream);
    }

private:
    ccl::comm comm;
    ccl::stream stream;
};


/*
 * Static helper functions 
 */
static int MPIR_OneCCLRedOpIsSupported(MPI_Op op)
{
    switch (op) {
    case MPI_SUM:
    case MPI_PROD:
    case MPI_MIN:
    case MPI_MAX:
        return 1;
    default:
        return 0;
    }
}

static ccl::reduction MPIR_OneCCLGetRedOp(MPI_Op op)
{
    switch (op) {
    case MPI_SUM:
        return ccl::reduction::sum;
    case MPI_PROD:
        return ccl::reduction::prod;
    case MPI_MIN:
        return ccl::reduction::min;
    case MPI_MAX:
        return ccl::reduction::max;
    default:
        return ccl::reduction::custom;
    }
}

static int MPIR_OneCCLDatatypeIsSupported(MPI_Datatype dtype)
{
    switch (MPIR_DATATYPE_GET_RAW_INTERNAL(dtype)) {
    case MPIR_INT8:
    case MPIR_UINT8:
    case MPIR_INT16:
    case MPIR_UINT16:  
    case MPIR_INT32:
    case MPIR_UINT32:
    case MPIR_INT64:
    case MPIR_UINT64:
    case MPIR_FLOAT16:
    case MPIR_FLOAT32:
    case MPIR_FLOAT64:
        return 1;
    default:
        return 0;
    }
}

static ccl::datatype MPIR_OneCCLGetDatatype(MPI_Datatype dtype)
{
    switch (MPIR_DATATYPE_GET_RAW_INTERNAL(dtype)) {
    case MPIR_INT8:
        return ccl::datatype::int8;
    case MPIR_UINT8:
        return ccl::datatype::uint8;
    case MPIR_INT16:
        return ccl::datatype::int16;
    case MPIR_UINT16:
        return ccl::datatype::uint16;
    case MPIR_INT32:
        return ccl::datatype::int32;
    case MPIR_UINT32:
        return ccl::datatype::uint32;
    case MPIR_INT64:
        return ccl::datatype::int64;
    case MPIR_UINT64:
        return ccl::datatype::uint64;
    case MPIR_FLOAT16:
        return ccl::datatype::float16;
    case MPIR_FLOAT32:
        return ccl::datatype::float32;
    case MPIR_FLOAT64:
        return ccl::datatype::float64;
    default:
        return -1;
    }
}

static int MPIR_OneCCLCheckInitAndInit(MPIR_Comm *comm_ptr, int rank)
{
    int mpi_errno = MPI_SUCCESS;

    if(!comm_ptr->cclcomm) {
        mpi_errno = MPIR_CCLcomm_init(comm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

    if (!comm_ptr->cclcomm->onecclcomm) {
        MPIR_OneCCLcomm* onecclcomm = new MPIR_OneCCLcomm(rank);
        comm_ptr->cclcomm->onecclcomm = static_cast<void*>(onecclcomm);
    }

fn_exit:
    return mpi_errno;
fn_fail:
    goto fn_exit;
}


/*
 * C-compatible public functions
 */
extern "C" {

    int MPIR_OneCCL_check_requirements_red_op(const void *sendbuf, void *recvbuf, MPI_Datatype datatype, MPI_Op op)
    {
        if (!MPIR_OneCCLRedOpIsSupported(op) || !MPIR_OneCCLDatatypeIsSupported(datatype) || !MPIR_CCL_check_both_gpu_bufs(sendbuf, recvbuf)) {
            return 0;
        }

        return 1;
    }

    int MPIR_OneCCL_Allreduce(const void *sendbuf, void *recvbuf, MPI_Aint count, MPI_Datatype datatype,
                            MPI_Op op, MPIR_Comm *comm_ptr, MPIR_Errflag_t errflag)
    {
        /* Check if the OneCCL comm is inited, and init it if not */
        MPIR_OneCCLCheckInitAndInit(comm_ptr, rank);

        /* Get the OneCCL reduction operation/datatype */
        ccl::reduction cclRedOp = MPIR_OneCCLGetRedOp(op);
        ccl::datatype cclDtype = MPIR_OneCCLGetDatatype(datatype);

        /* Handle in-place operations (OneCCL does not define an "in-place" value) */
        if (sendbuf == MPI_IN_PLACE) {
            sendbuf = recvbuf;
        }

        /* Make the OneCCL call */
        ccl::event e = MPIR_OneCCLcomm::allreduce(sendbuf, recvbuf, count, datatype, op, comm_ptr, errflag);
        e.wait();

        return MPI_SUCCESS;
    }

    int MPIR_OneCCLcomm_free(MPIR_Comm *comm){
        int mpi_errno = MPI_SUCCESS;

        MPIR_Assert(comm->cclcomm->onecclcomm);
        MPIR_OneCCLcomm* onecclcomm = static_cast<MPIR_OneCCLcomm*>(omm->cclcomm->onecclcomm);

        onecclcomm->~MPIR_OneCCLcomm();
        delete onecclcomm;
    }
}

#endif /* ENABLE_ONECCL */