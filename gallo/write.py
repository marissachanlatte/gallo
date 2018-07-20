from pathlib import Path

from gallo import fe

def _ele_header(num_elts):
    return "{} 3 1\n".format(num_elts)

def write_ele(ele_file, elts_list):
    ele_file = Path(ele_file)
    with ele_file.open("w") as ef:
        ef.write(_ele_header(len(elts_list)))
        for elt in elts_list:
            ef.write(str(elt))
