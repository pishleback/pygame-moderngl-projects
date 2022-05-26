import os


#special characters are < > /
#type // /< and /> to use these in text


class Tag():
    @staticmethod
    def Load(tags_string):        
        assert tags_string[0] == "<" and tags_string[-1] == ">"
        for a, letter in enumerate(tags_string):
            if letter == ">":
                break
        for b, letter in enumerate(reversed(tags_string)):
            if letter == "<":
                break
        b = len(tags_string) - b - 1
        assert a < b
        assert tags_string[b + 1] == "/"
        assert (name := tags_string[1:a]) == tags_string[b+2:-1]
        assert len(name) > 0
        
        sub_string = tags_string[len(name) + 2 : -len(name) - 3].strip()

        contents_string = ""
        subtag_strings = []
        i = 0
        in_esc = False
        tag_depth = 0
        while i < len(sub_string):
            letter = sub_string[i]
            if in_esc == False and letter == "/":
                in_esc = True
                if tag_depth != 0:
                    subtag_strings[-1] += letter
            else:
                if tag_depth == 0:
                    if in_esc:
                        contents_string += letter
                        in_esc = False
                    else:
                        if letter == ">":
                            raise Exception()
                        elif letter == "/":
                            raise Exception()
                        elif letter == "<":
                            tag_depth += 1
                            subtag_strings.append("<")
                        else:
                            contents_string += letter
                else:
                    subtag_strings[-1] += letter
                    if letter == "<":
                        tag_depth += 1
                    elif letter == ">" and in_esc:
                        tag_depth -= 2
                        in_esc = False
                
            i += 1
            
        assert in_esc == False
        assert tag_depth == 0

        sub_tags = [Tag.Load(string) for string in subtag_strings]

        return Tag(name, "\n".join([s for s in [s.strip() for s in contents_string.split("\n")] if s != ""]), sub_tags)
        
    def __init__(self, name, string, sub_tags):
        assert type(name) == str
        assert type(string) == str
        for tag in sub_tags:
            assert type(tag) == Tag
        self.name = name
        self.string = string
        self.sub_tags = sub_tags

    def save(self):
        if len(self.sub_tags) == 0:
            return "<" + self.name + ">" + self.string.replace("/", "//").replace("<", "/<").replace(">", "/>") + "</" + self.name + ">"
        else:
            string = self.string.replace("/", "//").replace("<", "/<").replace(">", "/>")
            lines = []
            lines.append("<" + self.name + ">")
            if len(self.string) > 0:
                lines.append("\t" + string.replace("\n", "\n\t"))
            lines.extend(["\t" + sub_tag.save().replace("\n", "\n\t") for sub_tag in self.sub_tags])
            lines.append("</" + self.name + ">")
            return "\n".join(lines)

    def get_sub_tags(self, name):
        return [tag for tag in self.sub_tags if tag.name == name]

    def get_sub_tag(self, name):
        sub_tags = self.get_sub_tags(name)
        if len(sub_tags) == 1:
            return sub_tags[0]
        else:
            raise Exception("Multiple or no tags with name " + name + " found!")

    def get_sub_strs(self, name):
        tags = self.get_sub_tags(name)
        for tag in tags:
            assert len(tag.sub_tags) == 0
        return [tag.string for tag in tags]

    def get_sub_str(self, name):
        strings = self.get_sub_strs(name)
        if len(strings) == 0:
            return None
        elif len(strings) == 1:
            return strings[0]
        else:
            raise Exception()




if __name__ == "__main__":
    f = open("oof.txt", "r")
    string = f.read()

##    for i in range(1, len(string)):
##        if string[i] == "?":
##            print(string[i - 10:i + 10])
    
    string = Tag.Load(string).save()
    print(string)




























    
