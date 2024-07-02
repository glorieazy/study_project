import json
import pprint

class asConfig:
    
    def __init__(self, filename=None):
        """
        read json from file
        """
        if filename is None:
            # construct a dummy class 
            self._data = None
        else:
            self.filename = filename
            with open( self.filename, "r" ) as jsonfile:
                self._data = json.load(jsonfile)
                print("Reading json from file is successful ...")
                jsonfile.close()


    def get_value(self, section: str, key: str ):        
        return self._data[section][key]

    def set_value(self, section: str, key: str, value ):        
        self._data[section][key] = value


    def write_config( self ):
        """
        writes json into file        
        """
        if self.filename is not None:        
            with open( self.filename, 'w', encoding='utf-8') as jsonfile:
                json.dump( self._data, jsonfile, ensure_ascii=False, indent=4)
                jsonfile.close()


    def print_config( self ):
        """
        prints data_json
        """     
        print()        
        print("----- Prints the config json-file ")      
        #print( json.dumps( self._data, indent=4) )
        pprint.pprint( self._data )
        print()        


#######################################
### Examples of working with config ###
#######################################
# config_json.print_config()
# value = config_json.get_value("dirs","media") 
# config_json.set_value("dirs","comments", "my nice dirs comments")
# config_json.write_config()

