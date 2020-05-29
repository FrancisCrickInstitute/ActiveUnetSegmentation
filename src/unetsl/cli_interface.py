# -*- coding: utf-8 -*-
import urwid
import json
import collections
import sys
import os

import readline, rlcompleter
import pathlib


class FileCompleter():
    """
        For getting a file from the command line. Searchs the path relative
        to a piece of text and offers suggestions. Uses readline.
    """
    def __init__(self):
        self.cwd = pathlib.Path(".")
        self.choices = [ p for p in self.cwd.iterdir()]
        self.last = ""
        self.suggestions = [str(p) for p in self.cwd.iterdir()] 
        
    def __call__(self, text, state):
        emsg = None
        if not text == self.last:
            try:
                self.getNewSuggestions(text, state)
            except:
                self.suggestions = []
                emsg = "failed to update: %s"%(sys.exc_info(),)
            if False:
                """
                    Using print to debug this is impossible. log to a file.
                    #TODO remove this.
                
                """
                with open("junk-log.txt", 'a', encoding="utf8") as f:
                    f.write(">>%s::%s\n"%(state, text))
                    if emsg:
                        f.write("!!%s\n"%emsg)
                    for sug in self.suggestions:
                        f.write("\t%s\n"%sug)
        if state < len(self.suggestions):
            return self.suggestions[state]
        return None
    
    def getNewSuggestions(self, raw_text, state):
        """
            Populets the self.suggestions field with suggestions.
            
            Args:
            raw_text: textused for finding the path
            state: unused, from rlcomplete interface.
        """
        text = os.path.expandvars(raw_text)
        chk = pathlib.Path(text).expanduser()
        
        if chk.is_dir() and text.endswith("/"):
            if chk!=self.cwd:
                self.cwd = chk
                self.choices = [ p for p in self.cwd.iterdir()]
            self.suggestions = [text] + [str(p) for p in self.choices]
            
        else:
            if chk.parent!=self.cwd:
                self.cwd = chk.parent
                self.choices = [ p for p in self.cwd.iterdir()]
            self.suggestions = [str(c) for c in self.choices if c.name.startswith(chk.name)]
        self.last = raw_text
        
        
class LayerCompleter():
    """
        For getting a layer name from the command line.
    """
    def __init__(self, layer_names):
        
        self.choices = layer_names
        self.last = ""
        self.suggestions = layer_names[:]
    def __call__(self, text, state):
        emsg = None
        try:
            self.getNewSuggestions(text, state)
        except:
            self.suggestions = []
            emsg = "failed to update: %s"%(sys.exc_info(),)
        if False:
            """
                Using print to debug this is impossible. log to a file.
                #TODO remove this.
            
            """
            with open("junk-log.txt", 'a', encoding="utf8") as f:
                f.write(">>%s::%s\n"%(state, text))
                if emsg:
                    f.write("!!%s\n"%emsg)
                for sug in self.suggestions:
                    f.write("\t%s\n"%sug)
        if state < len(self.suggestions):
            return self.suggestions[state]
        return None
    
    def getNewSuggestions(self, raw_text, state):
        """
            Populets the self.suggestions field with suggestions.
            
            Args:
            raw_text: textused for finding the path
            state: unused, from rlcomplete interface.
        """
        if len(raw_text) > 0:
            self.suggestions = [i for i in self.choices if i.startswith(raw_text)]
        else:
            self.suggestions = [self.choices[:]]

def getLayersPrompt(layer_names, message="select layer: "):
    readline.set_completer(LayerCompleter(layer_names))
    readline.set_completer_delims("")
    readline.parse_and_bind("tab: complete")
    return input(message)

def getFilePrompt(message):
    """
        Uses input to request a file by setting up readline with file co
    """
    readline.set_completer(FileCompleter())
    readline.set_completer_delims("")
    readline.parse_and_bind("tab: complete")
    return input(message)

def showError(input_widget):
    """
        Not currently implemented, used for debugging.
    """
    pass


class ParameterMenu:
    """
        Lists all of the parameters in parallel columns, working on making 
        sub-menus.
    """
    def __init__(self, config, title="Generic Config", verify={}, finish="finished"):
        self.validations = collections.defaultdict(lambda : lambda x: True, verify)
        self.config = config
        self.title = urwid.Text(title)
        self.status = urwid.Text("")
        self.cancelled = False
        self.finish = finish
        
    def createMenu(self):
        self.inputs = {}
        keys = []
        values = []
        rows = []
        for k,v in self.config.items():
            if isinstance(v, dict):
                print("found a dictionary")
            ed = urwid.Edit("","%s"%json.dumps(v, indent="  "))
            row = urwid.Columns([urwid.Text(k), ed])
            urwid.connect_signal(ed, 'change', self.onTextChange)
            self.inputs[k] = ed
            rows.append(row)
        
        
        
        can = urwid.Button("cancel")
        fin = urwid.Button(self.finish)
        self.finish = fin
        
        urwid.connect_signal(fin, 'click', self.onFinishedClicked)
        urwid.connect_signal(can, 'click', self.onCancelClicked)
        
        #keysPile = urwid.Pile(keys)
        #valuesPile = urwid.Pile(values)
        #columns = urwid.Columns([keysPile, valuesPile])
        columns = urwid.Pile(rows)
        buttons = urwid.Columns([urwid.Text(""), can, fin]);
        self.columns = columns
        self.buttons = buttons
        self.listed = urwid.Pile([columns, buttons])
        fill = urwid.Filler(self.listed)
        self.frame = urwid.Frame(fill, header=self.title, footer=self.status)
        
    def display(self):
        self.createMenu()
        self.loop = urwid.MainLoop(self.frame, unhandled_input=self.unhandled_input)
        self.loop.run()            
        
    def updateConfig(self):
        for k in self.config.keys():
            new_value = json.loads(self.inputs[k].get_text()[0])
            
            if(self.validations[k](new_value)):
                self.config[k] = new_value
            else:
                showError(self.inputs[k])
            
    def onFinishedClicked(self, item):
        self.status.set_text("clicked finish %s"%dir(item))
        raise urwid.ExitMainLoop()
    def onCancelClicked(self, item):
        self.cancelled = True
        raise urwid.ExitMainLoop()
        
    def onTextChange(self, item, item_text):
        self.status.set_text("i: %s, v: %s"%(str(item), str(item_text)))

    def unhandled_input(self, inp):
        if str(inp)=='tab':
            self.listed.set_focus(self.buttons)
            self.buttons.set_focus(self.finish)
        self.status.set_text("missed: %s"%repr(inp))
    
    
def configure(config, title = "update settings", finish="finished"):
    """
        Creates a ParameterMenu for updating a dict like object. the values 
        are entered as json.
    """
    menu = ParameterMenu(config, title, finish=finish)
    menu.display()
    if(menu.cancelled):
        return False
    else:
        menu.updateConfig()
        return True
    
    

if __name__=="__main__":
    config = json.load(sys.argv[1])
    
    configure(config)
