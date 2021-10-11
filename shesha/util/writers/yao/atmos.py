import numpy as np

def write_atm(file_name, atm, screen_file):
    """Write (append) atmospheric parameters to file for YAO use

    Args:
        file_name : (str) : name of the file to append the parameter to

        atm : (Param_atmos) : compass atmospheric parameters

        screen_file : (str) : path to the yao turbulent screen files
    """
    f = open(file_name,"a+")
    f.write("\n\n//------------------------------")
    f.write("\n//ATM  parameters")
    f.write("\n//------------------------------")

    f.write("\nr0              =" + str(atm.r0) + "; //qt 500 nm")
    f.write("\natm.dr0at05mic  = tel.diam/r0;")

    indexList = '"1"'
    for i in range(2, atm.nscreens + 1):
        indexList += ',"' + str(i) + '"'
    f.write("\natm.screen = &(\"" + screen_file+"\"+["+indexList + \
            "]+\".fits\")")
    f.write("\natm.layerspeed  = &(" + np.array2string(atm.windspeed, \
            separator=',', max_line_width=300) + ");")
    f.write("\natm.layeralt    = &(" + np.array2string(atm.alt, \
            separator=',', max_line_width=300) + ");")
    f.write("\natm.layerfrac   = &(" + np.array2string(atm.frac, \
            separator=',', max_line_width=300) + ");")
    f.write("\natm.winddir     = &(" + np.array2string(atm.winddir, \
            separator=',', max_line_width=300) + ");")
    f.close()

