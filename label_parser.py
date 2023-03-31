import xml.sax
from dataclasses import dataclass


@dataclass
class Label():
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class LabelHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.folder = None
        self.filename = None
        self.width = None
        self.height = None
        self.current_tag = None
        self.objects = []
        self.current_object = None

    def startElement(self, tag, attributes):
        if tag == "object":
            if self.current_object is None:
                self.current_object = 0
            else:
                self.current_object += 1
            self.objects.append(Label("", 0, 0, 0, 0))
        elif tag in ["xmin", "xmax", "ymin", "ymax", "name", "folder", "filename", "width", "height"]:
            self.current_tag = tag

    def endElement(self, tag):
        if tag in ["xmin", "xmax", "ymin", "ymax", "name", "folder", "filename", "width", "height"]:
            self.current_tag = None

    def characters(self, content):
        if self.current_tag in ["xmin", "xmax", "ymin", "ymax", "width", "height"]:
            parsed_content = int(content)
            if self.current_tag == "xmin":
                self.objects[self.current_object].xmin = parsed_content
            elif self.current_tag == "ymin":
                self.objects[self.current_object].ymin = parsed_content
            elif self.current_tag == "xmax":
                self.objects[self.current_object].xmax = parsed_content
            elif self.current_tag == "ymax":
                self.objects[self.current_object].ymax = parsed_content
            elif self.current_tag == "width":
            	self.width = parsed_content
            elif self.current_tag == "height":
            	self.height = parsed_content
        elif self.current_tag == "name":
            self.objects[self.current_object].name = content
        elif self.current_tag == "folder":
            self.folder = content
        elif self.current_tag == "filename":
            self.filename = content


def parse_xml_file(fname):
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)
	handler = LabelHandler()
	parser.setContentHandler(handler)
	parser.parse(fname)
	return handler.folder, handler.filename, handler.width, handler.height, handler.objects
