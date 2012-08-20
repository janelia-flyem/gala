import argparse
import os
import sys
import datetime
import getpass

import app_logger, option_manager 

class Session:
    def __init__(self, name, description, master_logger, option_fn=None):
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

        # create json and command line options 
        self.options_parser = option_manager.OptionManager(master_logger, parser)

        if option_fn is not None:
            option_fn(self.options_parser)
        parser.epilog = self.options_parser.help_message()

        args = parser.parse_args()

        self.verbose = args.verbose
        self.session_location = args.session_location
        self.config_file = args.config_file

        # set verbosity 
        if self.verbose:
            app_logger.set_debug_console()

        if not os.path.exists(self.session_location):
            os.makedirs(self.session_location)

        # check for old files 
        if os.path.exists(self.session_location + "/.running"):
            raise Exception("Session " + args.output_dir + " incomplete")
        if os.path.exists(self.session_location + "/.finished"):
            master_logger.warning("Overwriting previous session: " + self.session_location)
            os.remove(self.session_location + "/.finished")

        # set log name
        log_filename = self.session_location + "/." + name + ".log"
        app_logger.set_log_file(master_logger, log_filename)

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
