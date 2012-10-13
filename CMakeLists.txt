CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (ray)

include (ExternalProject)

set (RUN_ENVIRONMENT "Workstation" CACHE TYPE STRING)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif ()


################################################################################
# Check if FLYEM_BUILD_DIR has already been assigned.  If not, create a default.
set (FLYEM_BUILD_DIR "None" CACHE TYPE STRING)

if (${FLYEM_BUILD_DIR} STREQUAL "None")
    message (FATAL_ERROR "ERROR: FlyEM build directory (for all downloads & builds) should be specified via -DFLYEM_BUILD_DIR=<path> on cmake command line.")
endif ()

message ("FlyEM downloads and builds will be placed here: ${FLYEM_BUILD_DIR}")

###############################################################################

###############################################################################
# Download and install flyem-build, if it isn't already in FLYEM_BUILD_DIR.
set (FLYEM_BUILD_REPO_DIR ${FLYEM_BUILD_DIR}/src/flyem-build)
if (NOT EXISTS ${FLYEM_BUILD_REPO_DIR}/python.cmake)
    message ("Installing flyem-build repo...")
    ExternalProject_Add(flyem-build
        PREFIX              ${FLYEM_BUILD_DIR}
        GIT_REPOSITORY      https://github.com/janelia-flyem/flyem-build.git
        UPDATE_COMMAND      ""
        PATCH_COMMAND       ""
        CONFIGURE_COMMAND   "" 
        BUILD_COMMAND       ""
        BUILD_IN_SOURCE     1
        INSTALL_COMMAND     ""
    )
    message ("\n**********************************************************\n")
    message ("\nAfter running make, you must re-run the cmake command once")
    message ("flyem-build has been downloaded!\n")
    message ("\n***********************************************************\n")
else ()
    ###############################################################################

    # Use modules from the downloaded flyem-build
    set (CMAKE_MODULE_PATH ${FLYEM_BUILD_REPO_DIR})
    message("Using cmake modules from ${FLYEM_BUILD_REPO_DIR}")

    # Download and compile dependencies
    include (cython)
    include (h5py)
    include (numpy)
    include (scipy)
    include (vigra)
    include (pil)

    include (EasyInstall)
    easy_install (networkx)
    easy_install (progressbar)
    easy_install (scikits-image)
    easy_install (scikit-learn)

    # Install ray
    add_custom_target (ray ALL 
        DEPENDS     ${APP_DEPENDENCIES}
        COMMENT     "Building ray and all dependencies...")
    
    ###############################################################################
endif()
