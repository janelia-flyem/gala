CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (gala)

include (ExternalProject)

set (RUN_ENVIRONMENT "Workstation" CACHE TYPE STRING)
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif ()


################################################################################
# Check if FLYEM_BUILD_DIR has already been assigned.  If not, create a default.
set (BUILDEM_DIR "None" CACHE TYPE STRING)

if (${BUILDEM_DIR} STREQUAL "None")
    message (FATAL_ERROR "ERROR: Buildem directory (for all downloads & builds) should be specified via -DBUILDEM_DIR=<path> on cmake command line.")
endif ()

message ("FlyEM downloads and builds will be placed here: ${BUILDEM_DIR}")

###############################################################################

###############################################################################
# Download and install flyem-build, if it isn't already in FLYEM_BUILD_DIR.
set (BUILDEM_REPO_DIR ${BUILDEM_DIR}/src/buildem)
if (NOT EXISTS ${BUILDEM_REPO_DIR}/python.cmake)
    message ("Installing buildem repo...")
    ExternalProject_Add(buildem
        PREFIX              ${BUILDEM_DIR}
        GIT_REPOSITORY      http://github.com/janelia-flyem/buildem.git
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
    set (CMAKE_MODULE_PATH ${BUILDEM_REPO_DIR})
    message("Using cmake modules from ${BUILDEM_REPO_DIR}")
    include (BuildSupport)

    # Download and compile dependencies
    include (python)
    include (pyopengl)
    include (scipy)
    include (h5py)
    include (pil)
    include (ilastik)

    include (networkx)
    include (progressbar)
    include (matplotlib)
    include (scikit-image)
    include (scikit-learn)
    include (syngeo)
    include (raveler-utils)

    # Install gala
    add_custom_target (gala ALL
        DEPENDS     ${APP_DEPENDENCIES}
        COMMAND     ${BUILDEM_ENV_STRING} python setup.py install
        WORKING_DIRECTORY   ${CMAKE_CURRENT_SOURCE_DIR})
    
    # Once all dependencies have built, calling "make gala-only" will make
    # sure you are only running the python setup.py install.
    add_custom_target (gala-only
        COMMAND     ${BUILDEM_ENV_STRING} python setup.py install
        WORKING_DIRECTORY   ${CMAKE_CURRENT_SOURCE_DIR})
    
    ###############################################################################
endif()
