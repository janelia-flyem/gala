#!/usr/bin/env python
#
# Copyright 2012 HHMI.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of HHMI nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import argparse
import os
import sys
import datetime
import getpass

import option_manager 

class Session:
    """The Session Manager 

    Encapsulates logging and command-line options for session location
    and other parameters pertinent to sessions.  Also creates 
    placeholder files (.running and .finished) in session location.
    """
    def __init__(self, name, description, master_logger, applogger, option_fn=None):
        self.master_logger = master_logger
        
        self.session_location = None
        self.start_time = str(datetime.datetime.now())
        self.end_time = None
        self.options_parser = None       
 
        parser = argparse.ArgumentParser(description=description, 
            formatter_class=argparse.RawDescriptionHelpFormatter)
        
        parser.add_argument('session_location', type=str, help="Directory for session")
        parser.add_argument('--config-file', '-c', help='json config file') 
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
            help='Print runtime information about execution.')
        parser.add_argument('--regression', action="store_true", default=False, 
                help=argparse.SUPPRESS) 

        # create json and command line options 
        self.options_parser = option_manager.OptionManager(master_logger, parser)

        if option_fn is not None:
            option_fn(self.options_parser)
        parser.epilog = self.options_parser.help_message()

        args = parser.parse_args()

        self.regression = args.regression
        self.verbose = args.verbose
        self.session_location = args.session_location
        self.config_file = args.config_file

        # set verbosity 
        if self.verbose:
            applogger.set_debug_console()

        if not os.path.exists(self.session_location):
            os.makedirs(self.session_location)

        # check for old files 
        if os.path.exists(self.session_location + "/.running"):
            raise Exception("Session " + self.session_location + " incomplete")
        if os.path.exists(self.session_location + "/.finished"):
            master_logger.warning("Overwriting previous session: " + self.session_location)
            os.remove(self.session_location + "/.finished")

        # set log name
        log_filename = self.session_location + "/." + name + ".log"
        applogger.set_log_file(log_filename, self.regression)

        # load the options from the config file and args    
        self.options = self.options_parser.load_config(self.config_file, args)
        self.master_logger.info("Options loaded and verified")
        

        self.export_config()
            
        with open(self.session_location + "/.running", 'w'):
            pass

    def export_config(self):
        config_data = {}
        meta_data = {}
        meta_data["version"] = 1.0
        meta_data["user"] = getpass.getuser()
        meta_data["issued-command"] = ' '.join(sys.argv)
        meta_data["issued-from"] = os.path.realpath('.')
        meta_data["start-time"] = self.start_time 
        if self.end_time:
            meta_data["end-time"] = self.end_time 
        config_data["meta-data"] = meta_data
        
        if self.options_parser is not None:
            self.options_parser.export_json(self.session_location + "/config.json", config_data)


    def __del__(self):
        if self.session_location is not None:
            if os.path.exists(self.session_location + "/.running"):
                os.remove(self.session_location + "/.running")
            with open(self.session_location + "/.finished", 'w'):
                pass
            self.end_time = str(datetime.datetime.now())
            self.export_config()
