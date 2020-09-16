## Module to help incorporate active documentation into your work framework. Initialize documents, embed images, create tables, etc. 
import pathlib
import __main__
import os
import datetime
from mdutils.mdutils import MdUtils
from mdutils import Html

pathname = "../docs/script_docs/"

def initialize_doc():
    """Initializes a markdown document in docs/script_docs named after this script with a correctly updated date. 

    :returns: a mdutils mdfile object. Must be written (invoke `mdfile.create_md_file`) in order to manifest a real file. 
    """
    filename = pathlib.Path(__main__.__file__).stem 
    mdFile = MdUtils(file_name = os.path.join(pathname,filename),title = "Script documentation for file: "+ filename+", Updated on:" +str(datetime.datetime.now()))
    return mdFile

def insert_image(md,path,size = None,align = None):
    """Inserts an image at a new line in the markdown document. Can do basic image formatting as well: size can be specified as a list [width, height], and/or aligned. 

    :param md: mdfile object. 
    :param path: the path to an image file. 
    :param size: a list specifying the width and height of the new array. One entry can be none to just change one aspect of the image.  
    :param align: provides alignment according to Html.image specifications. i.e. 'center'
    """
    ## parse arguments:
    if size:
        assert type(size) is list, "size must be a list"
        assert len(size) == 2, "size must have two elements (width and height)"
        assert not all([s is None for s in size]), "one entry must be non-none"
        if size[0] is None:
            size = "x{}".format(size[1])
        elif size[1] is None:
            size = str(size[0])
        else:
            size = "{w}x{h}".format(w=size[0],h=size[1])
    if align:
        assert type(align) is str, "align argument must be string."

    md.new_line(Html.image(path=path, size=size, align=align))
