/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef PMI2_H_INCLUDED
#define PMI2_H_INCLUDED

#ifndef PMI_VERSION
#define PMI_VERSION    2
#define PMI_SUBVERSION 0
#endif

#define PMI2_MAX_KEYLEN 64
#define PMI2_MAX_VALLEN 1024
#define PMI2_MAX_ATTRVALUE 1024
#define PMI2_ID_NULL -1

#if defined(__cplusplus)
extern "C" {
#endif

/* WARNING: PMI 2 is DEPRECATED. New features will be developed in PMI 1. */

/*D
PMI2_CONSTANTS - PMI2 definitions

Error Codes:
+ PMI2_SUCCESS - operation completed successfully
. PMI2_FAIL - operation failed
. PMI2_ERR_NOMEM - input buffer not large enough
. PMI2_ERR_INIT - PMI not initialized
. PMI2_ERR_INVALID_ARG - invalid argument
. PMI2_ERR_INVALID_KEY - invalid key argument
. PMI2_ERR_INVALID_KEY_LENGTH - invalid key length argument
. PMI2_ERR_INVALID_VAL - invalid val argument
. PMI2_ERR_INVALID_VAL_LENGTH - invalid val length argument
. PMI2_ERR_INVALID_LENGTH - invalid length argument
. PMI2_ERR_INVALID_NUM_ARGS - invalid number of arguments
. PMI2_ERR_INVALID_ARGS - invalid args argument
. PMI2_ERR_INVALID_NUM_PARSED - invalid num_parsed length argument
. PMI2_ERR_INVALID_KEYVALP - invalid keyvalp argument
. PMI2_ERR_INVALID_SIZE - invalid size argument
- PMI2_ERR_OTHER - other unspecified error

D*/
#define PMI2_SUCCESS                0
#define PMI2_FAIL                   -1
#define PMI2_ERR_INIT               1
#define PMI2_ERR_NOMEM              2
#define PMI2_ERR_INVALID_ARG        3
#define PMI2_ERR_INVALID_KEY        4
#define PMI2_ERR_INVALID_KEY_LENGTH 5
#define PMI2_ERR_INVALID_VAL        6
#define PMI2_ERR_INVALID_VAL_LENGTH 7
#define PMI2_ERR_INVALID_LENGTH     8
#define PMI2_ERR_INVALID_NUM_ARGS   9
#define PMI2_ERR_INVALID_ARGS       10
#define PMI2_ERR_INVALID_NUM_PARSED 11
#define PMI2_ERR_INVALID_KEYVALP    12
#define PMI2_ERR_INVALID_SIZE       13
#define PMI2_ERR_OTHER              14

/*S
PMI2_keyval_t - keyval structure used by PMI2_Spawn_multiple

Fields:
+ key - name of the key
- val - value of the key

S*/
    typedef struct PMI2_keyval_t {
        const char *key;
        const char *val;
    } PMI2_keyval_t;

/*@
  PMI2_Connect_comm_t - connection structure used when connecting to other jobs

  Fields:
  + read - Read from a connection to the leader of the job to which
    this process will be connecting. Returns 0 on success or an MPI
    error code on failure.
  . write - Write to a connection to the leader of the job to which
    this process will be connecting. Returns 0 on success or an MPI
    error code on failure.
  . ctx - An anonymous pointer to data that may be used by the read
    and write members.
  - isMain - Indicates which process is "main"; may have the values
    1 (is main), 0 (is not main), or -1 (neither is designated as
    main). The two processes must agree on which process is main, or
    both must select -1 (neither is main).

  Notes:
  A typical implementation of these functions will use the read and
  write calls on a pre-established file descriptor (fd) between the
  two leading processes. This will be needed only if the PMI server
  cannot access the KVS spaces of another job (this may happen, for
  example, if each mpiexec creates the KVS spaces for the processes
  that it manages).

@*/
    typedef struct PMI2_Connect_comm {
        int (*read) (void *buf, int maxlen, void *ctx);
        int (*write) (const void *buf, int len, void *ctx);
        void *ctx;
        int isMain;
    } PMI2_Connect_comm_t;

/*@
  PMI2_Init - initialize the Process Manager Interface

  Output Parameter:
  + spawned - spawned flag
  . size - number of processes in the job
  . rank - rank of this process in the job
  - appnum - which executable is this on the mpiexec commandline

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  Initialize PMI for this process group. The value of spawned indicates whether
  this process was created by 'PMI2_Spawn_multiple'.  'spawned' will be non-zero
  iff this process group has a parent.

@*/
    int PMI2_Init(int *spawned, int *size, int *rank, int *appnum);

/*@
  PMI2_Finalize - finalize the Process Manager Interface

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  Finalize PMI for this job.

@*/
    int PMI2_Finalize(void);

/*@
  PMI2_Initialized - check if PMI has been initialized

  Return values:
  Non-zero if PMI2_Initialize has been called successfully, zero otherwise.

@*/
    int PMI2_Initialized(void);

/*@
  PMI2_Abort - abort the process group associated with this process

  Input Parameters:
  + flag - non-zero if all processes in this job should abort, zero otherwise
  - error_msg - error message to be printed

  Return values:
  If the abort succeeds this function will not return.  Returns an MPI
  error code otherwise.

@*/
    int PMI2_Abort(int flag, const char msg[]);

/*@
  PMI2_Spawn - spawn a new set of processes

  Input Parameters:
  + count - count of commands
  . cmds - array of command strings
  . argcs - size of argv arrays for each command string
  . argvs - array of argv arrays for each command string
  . maxprocs - array of maximum processes to spawn for each command string
  . info_keyval_sizes - array giving the number of elements in each of the
    'info_keyval_vectors'
  . info_keyval_vectors - array of keyval vector arrays
  . preput_keyval_size - Number of elements in 'preput_keyval_vector'
  . preput_keyval_vector - array of keyvals to be pre-put in the spawned keyval space
  - jobIdSize - size of the buffer provided in jobId

  Output Parameter:
  + jobId - job id of the spawned processes
  - errors - array of errors for each command

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  This function spawns a set of processes into a new job.  The 'count'
  field refers to the size of the array parameters - 'cmd', 'argvs', 'maxprocs',
  'info_keyval_sizes' and 'info_keyval_vectors'.  The 'preput_keyval_size' refers
  to the size of the 'preput_keyval_vector' array.  The 'preput_keyval_vector'
  contains keyval pairs that will be put in the keyval space of the newly
  created job before the processes are started.  The 'maxprocs' array
  specifies the desired number of processes to create for each 'cmd' string.
  The actual number of processes may be less than the numbers specified in
  maxprocs.  The acceptable number of processes spawned may be controlled by
  ``soft'' keyvals in the info arrays.  The ``soft'' option is specified by
  mpiexec in the MPI-2 standard.  Environment variables may be passed to the
  spawned processes through PMI implementation specific 'info_keyval' parameters.
@*/
    int PMI2_Job_Spawn(int count, const char *cmds[],
                       int argcs[], const char **argvs[],
                       const int maxprocs[],
                       const int info_keyval_sizes[],
                       const PMI2_keyval_t * info_keyval_vectors[],
                       int preput_keyval_size,
                       const PMI2_keyval_t preput_keyval_vector[],
                       char jobId[], int jobIdSize, int errors[]);


/*@
  PMI2_Job_GetId - get job id of this job

  Input parameters:
  . jobid_size - size of buffer provided in jobid

  Output parameters:
  . jobid - the job id of this job

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Job_GetId(char jobid[], int jobid_size);

/*@
  PMI2_Job_Connect - connect to the parallel job with ID jobid

  Input parameters:
  . jobid - job id of the job to connect to

  Output parameters:
  . conn - connection structure used to exteblish communication with
    the remote job

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  This just "registers" the other parallel job as part of a parallel
  program, and is used in the PMI2_KVS_xxx routines (see below). This
  is not a collective call and establishes a connection between all
  processes that are connected to the calling processes (on the one
  side) and that are connected to the named jobId on the other
  side. Processes that are already connected may call this routine.

@*/
    int PMI2_Job_Connect(const char jobid[], PMI2_Connect_comm_t * conn);

/*@
  PMI2_Job_Disconnect - disconnects from the job with ID jobid

  Input parameters:
  . jobid - job id of the job to connect to

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Job_Disconnect(const char jobid[]);

/*@
  PMI2_KVS_Put - put a key/value pair in the keyval space for this job

  Input Parameters:
  + key - key
  - value - value

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  If multiple PMI2_KVS_Put calls are made with the same key between
  calls to PMI2_KVS_Fence, the behavior is undefined. That is, the
  value returned by PMI2_KVS_Get for that key after the PMI2_KVS_Fence
  is not defined.

@*/
    int PMI2_KVS_Put(const char key[], const char value[]);
/*@
  PMI2_KVS_Fence - commit all PMI2_KVS_Put calls made before this fence

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  This is a collective call across the job.  All PMI2_KVS_Put operations
  performed by any process in the same job must be visible to all
  processes (by using PMI2_KVS_Get) after PMI2_KVS_Fence completes.
  However, a PMI implementation could make this a lazy operation by not
  waiting for all processes to enter their corresponding PMI2_KVS_Fence
  until some process issues a PMI2_KVS_Get. Thus PMI2_KVS_Fence alone
  may not serve as a barrier.  Using PMI2_KVS_GET for a non-existent key
  after PMI2_KVS_Fence will have the same effect as an barrier since
  PMI2_KVS_GET will not return until all processes posted PMI2_KVS_Fence,
  even though PMI2_KVS_Get will return error since the key does not
  exist.

@*/
    int PMI2_KVS_Fence(void);

/*@
  PMI2_KVS_Get - returns the value associated with key in the key-value
      space associated with the job ID jobid

  Input Parameters:
  + jobid - the job id identifying the key-value space in which to look
    for key.  If jobid is NULL, look in the key-value space of this job.
  . src_pmi_id - the pmi id of the process which put this key pair.  This
    is just a hint to the server.  PMI2_ID_NULL should be passed if no
    hint is provided.
  . key - key
  - maxvalue - size of the buffer provided in value

  Output Parameters:
  + value - value associated with key
  - vallen - length of the returned value, or, if the length is longer
    than maxvalue, the negative of the required length is returned

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_KVS_Get(const char *jobid, int src_pmi_id, const char key[], char value[],
                     int maxvalue, int *vallen);

/*@
  PMI2_Info_GetNodeAttr - returns the value of the attribute associated
      with this node

  Input Parameters:
  + name - name of the node attribute
  . valuelen - size of the buffer provided in value
  - waitfor - if non-zero, the function will not return until the
    attribute is available

  Output Parameters:
  + value - value of the attribute
  - found - non-zero indicates that the attribute was found

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  This provides a way, when combined with PMI2_Info_PutNodeAttr, for
  processes on the same node to share information without requiring a
  more general barrier across the entire job.

  If waitfor is non-zero, the function will never return with found
  set to zero.

  Predefined attributes:
  + memPoolType - If the process manager allocated a shared memory
    pool for the MPI processes in this job and on this node, return
    the type of that pool. Types include sysv, anonmmap and ntshm.
  . memSYSVid - Return the SYSV memory segment id if the memory pool
    type is sysv. Returned as a string.
  . memAnonMMAPfd - Return the FD of the anonymous mmap segment. The
    FD is returned as a string.
  - memNTName - Return the name of the Windows NT shared memory
    segment, file mapping object backed by system paging
    file.  Returned as a string.

@*/
    int PMI2_Info_GetNodeAttr(const char name[], char value[], int valuelen, int *found,
                              int waitfor);

/*@
  PMI2_Info_GetNodeAttrIntArray - returns the value of the attribute associated
      with this node.  The value must be an array of integers.

  Input Parameters:
  + name - name of the node attribute
  - arraylen - number of elements in array

  Output Parameters:
  + array - value of attribute
  . outlen - number of elements returned
  - found - non-zero if attribute was found

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  Notice that, unlike PMI2_Info_GetNodeAttr, this function does not
  have a waitfor parameter, and will return immediately with found=0
  if the attribute was not found.

  Predefined array attribute names:
  + localRanksCount - Return the number of local ranks that will be
    returned by the key localRanks.
  . localRanks - Return the ranks in MPI_COMM_WORLD of the processes
    that are running on this node.
  - cartCoords - Return the Cartesian coordinates of this process in
    the underlying network topology. The coordinates are indexed from
    zero. Value only if the Job attribute for physTopology includes
    cartesian.

@*/
    int PMI2_Info_GetNodeAttrIntArray(const char name[], int array[], int arraylen, int *outlen,
                                      int *found);

/*@
  PMI2_Info_PutNodeAttr - stores the value of the named attribute
  associated with this node

  Input Parameters:
  + name - name of the node attribute
  - value - the value of the attribute

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Notes:
  For example, it might be used to share segment ids with other
  processes on the same SMP node.

@*/
    int PMI2_Info_PutNodeAttr(const char name[], const char value[]);

/*@
  PMI2_Info_GetJobAttr - returns the value of the attribute associated
  with this job

  Input Parameters:
  + name - name of the job attribute
  - valuelen - size of the buffer provided in value

  Output Parameters:
  + value - value of the attribute
  - found - non-zero indicates that the attribute was found

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Info_GetJobAttr(const char name[], char value[], int valuelen, int *found);

/*@
  PMI2_Info_GetJobAttrIntArray - returns the value of the attribute associated
      with this job.  The value must be an array of integers.

  Input Parameters:
  + name - name of the job attribute
  - arraylen - number of elements in array

  Output Parameters:
  + array - value of attribute
  . outlen - number of elements returned
  - found - non-zero if attribute was found

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

  Predefined array attribute names:

  + universeSize - The size of the "universe" (defined for the MPI
    attribute MPI_UNIVERSE_SIZE

  . hasNameServ - The value hasNameServ is true if the PMI2 environment
    supports the name service operations (publish, lookup, and
    unpublish).

  . physTopology - Return the topology of the underlying network. The
    valid topology types include cartesian, hierarchical, complete,
    kautz, hypercube; additional types may be added as necessary. If
    the type is hierarchical, then additional attributes may be
    queried to determine the details of the topology. For example, a
    typical cluster has a hierarchical physical topology, consisting
    of two levels of complete networks - the switched Ethernet or
    Infiniband and the SMP nodes. Other systems, such as IBM BlueGene,
    have one level that is cartesian (and in virtual node mode, have a
    single-level physical topology).

  . physTopologyLevels - Return a string describing the topology type
    for each level of the underlying network. Only valid if the
    physTopology is hierarchical. The value is a comma-separated list
    of physical topology types (except for hierarchical). The levels
    are ordered starting at the top, with the network closest to the
    processes last. The lower level networks may connect only a subset
    of processes. For example, for a cartesian mesh of SMPs, the value
    is cartesian,complete. All processes are connected by the
    cartesian part of this, but for each complete network, only the
    processes on the same node are connected.

  . cartDims - Return a string of comma-separated values describing
    the dimensions of the Cartesian topology. This must be consistent
    with the value of cartCoords that may be returned by
    PMI2_Info_GetNodeAttrIntArray.

    These job attributes are just a start, but they provide both an
    example of the sort of external data that is available through the
    PMI interface and how extensions can be added within the same API
    and wire protocol. For example, adding more complex network
    topologies requires only adding new keys, not new routines.

  . isHeterogeneous - The value isHeterogeneous is true if the
    processes belonging to the job are running on nodes with different
    underlying data models.

@*/
    int PMI2_Info_GetJobAttrIntArray(const char name[], int array[], int arraylen, int *outlen,
                                     int *found);

/*@
  PMI2_Nameserv_publish - publish a name

  Input parameters:
  + service_name - string representing the service being published
  . info_ptr -
  - port - string representing the port on which to contact the service

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Nameserv_publish(const char service_name[], const PMI2_keyval_t * info_ptr,
                              const char port[]);

/*@
  PMI2_Nameserv_lookup - lookup a service by name

  Input parameters:
  + service_name - string representing the service being published
  . info_ptr -
  - portLen - size of buffer provided in port

  Output parameters:
  . port - string representing the port on which to contact the service

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Nameserv_lookup(const char service_name[], const PMI2_keyval_t * info_ptr,
                             char port[], int portLen);
/*@
  PMI2_Nameserv_unpublish - unpublish a name

  Input parameters:
  + service_name - string representing the service being unpublished
  - info_ptr -

  Return values:
  Returns 'MPI_SUCCESS' on success and an MPI error code on failure.

@*/
    int PMI2_Nameserv_unpublish(const char service_name[], const PMI2_keyval_t * info_ptr);



#if defined(__cplusplus)
}
#endif
#endif                          /* PMI2_H_INCLUDED */
