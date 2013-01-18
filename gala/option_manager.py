import argparse
import json

class OptionConfig:
    def __init__(self, description, default_val, required, dtype, warning, verify_fn):
        self.description = description
        self.default_val = default_val
        self.required = required
        self.dtype = dtype
        self.warning = warning
        self.verify_fn = verify_fn

class OptionNamespace:
    def __init__(self):
        pass
    def get_value(self, name):
        name = name.replace('-', '_')
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise Exception("Option attribute: " + name + " does not exist")

    def __contains__(self, key):
        key = key.replace('-', '_')
        return key in self.__dict__

class OptionManager:
    """
    Handles specification (through create_option) and parsing of options (through
    load_config).    By convention, hyphenated flags for arguments will set 
    underscored attributes of the options property.

    For example:
        "ilp-file" option creates an --ilp-file flag and if the dtype is 'str',
        we automatically create a property options.ilp_file
    """

    def __init__(self, master_logger, arg_parser):
        self.master_logger = master_logger
        self.options_config = {}
        self.options = OptionNamespace()
        self.arg_parser = arg_parser

    def load_config(self, file_name, args=None):
        """Returns an OptionNamespace"""
        
        json_data = None
        
        if file_name is None:
            self.master_logger.warning("Configuration file not supplied, using flags only")
        else:
            try:
                json_file = open(file_name)
                json_data = json.load(json_file)
                json_file.close()        
            except Exception, e:
                self.master_logger.warning("error in opening " + file_name + \
                    " (%s) , using flags only" % str(e))

        args_dict = None
        if args:
            args_dict = args.__dict__

        for name, option in self.options_config.items():
            option_val = None
            if json_data is not None and name in json_data:
                option_val = json_data[name]
 
            overridden = False
            #nametemp = name.replace('-','_') 
            nametemp = name
            if args_dict and nametemp in args_dict and args_dict[nametemp] is not None:
                option_val = args_dict[nametemp]
                overridden = True           

            if option.required and option_val is None:
                raise Exception(name + " needs to be specified")
            elif option.warning and option_val is None:
                self.master_logger.warning(name + " was not specified")
            elif option_val is None:
                self.master_logger.debug(name + " was not specified and set to default "
                    + str(option.default_val))
            elif option_val is not None and overridden:
                self.master_logger.debug(name + " was overridden by command line to "
                    + str(option_val))
            else:
                self.master_logger.debug(name + " was set to " + str(option_val))
               
            if option_val is not None:
                setattr(self.options, name.replace('-', '_'), option_val)
            else:
                setattr(self.options, name.replace('-', '_'), option.default_val)

        for name, option in self.options_config.items():
            if option.verify_fn is not None:
                option.verify_fn(self, self.options, self.master_logger)

        return self.options

    def verify_option(self, name):
        if name not in self.options_config:
            raise Exception("Trying to verify a non-existent option")
        self.options_config[name].verify_fn(self, self.options, self.master_logger)

    def help_message(self):
        config_format = "Command Options\n\n"
        for name, option in self.options_config.items():
            if option.dtype == bool:
                name = "[enable|disable]-" + name
            config_format += name + ": " + option.description + " "
            if option.required:
                config_format += "(required: " + str(option.dtype) + ")"
            else:
                config_format += "(default: " + str(option.default_val) + ")"
            config_format += "\n\n"
        return config_format

    def export_json(self, file_name, json_data=None):
        if json_data is None:
            json_data = {}

        for name, option in self.options_config.items():
            try:
                option_val = self.options.get_value(name)
                #if option_val is not None:
                json_data[name] = option_val
            except Exception:
                return

        fout = open(file_name, 'w')
        fout.write(json.dumps(json_data, indent=4))

    def create_option(self, unique_name, description, default_val=None,
            required=True, dtype=str, verify_fn=None, num_args=None, shortcut=None, 
            warning=False,  hidden=False):
        # create options
        if unique_name in self.options:
            raise Exception("Cannot create option with duplicate name: " + str(unique_name))

        option = OptionConfig(description, default_val, required, dtype, warning, verify_fn)
        
        self.options_config[unique_name] = option 

        help_message = description
        if hidden == True:
            help_message = argparse.SUPPRESS                

        if dtype == bool:
            if default_val is None:
                raise Exception("Cannot create a bool option without a default value: "
                    + str(unique_name))
            true_val = "store_true" 
            false_val = "store_false" 
           
            #group = self.arg_parser.add_mutuall_exclusive_group() 
            self.arg_parser.add_argument("--" + "enable-" + unique_name, action="store_true",
                    help=help_message, default=None, dest=unique_name) 
            self.arg_parser.add_argument("--" + "disable-" + unique_name, action="store_false",
                    help=help_message, default=None, dest=unique_name) 
        else:
            if shortcut:
                self.arg_parser.add_argument("--" + unique_name, "-" + shortcut, type=dtype, 
                    nargs=num_args, help=help_message, default=None, dest=unique_name) 
            else:
                self.arg_parser.add_argument("--" + unique_name, type=dtype, nargs=num_args, 
                    help=help_message, default=None, dest=unique_name) 

