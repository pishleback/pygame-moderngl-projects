import os
import random
import itertools
import fractions
from ftree import myxml
from ftree import treeview
import networkx as nx


PATH = None
COLOURS = {"bad" : [196, 128, 128],
           "neutral" : [128, 128, 196],
           "good" : [128, 196, 128],
           "bg" : [32, 32, 32],
           "text" : [255, 255, 255]}
SURFACE_TEXT_HEIGHT = 70


def STRING_EDITOR_TEXT_COLOUR_GETTER(selected, string):                    
    if selected:
        if string == "":
            return [0, 128, 0]
        else:
            return [0, 255, 0]
    else:
        if string == "":
            return [128, 128, 128]
        else:
            return None


class Info():
    @staticmethod
    def LoadInfos(tag):
        assert tag.name == "infos"
        kinds = {"date" : Date, "subinfo" : SubInfo, "string" : String}
        for sub_tag in tag.sub_tags:
            assert sub_tag.name in kinds
        return [kinds[sub_tag.name].Load(sub_tag) for sub_tag in tag.sub_tags]
    
    @staticmethod
    def SaveInfos(infos):
        for info in infos:
            assert issubclass(type(info), Info)
        return myxml.Tag("infos", "", [info.save() for info in infos])

        
    def __init__(self):
        pass




class String(Info):
    @staticmethod
    def Load(tag):
        assert tag.name == "string"
        assert len(tag.sub_tags) == 0
        return String(tag.string)
    
    def __init__(self, string):
        Info.__init__(self)
        assert type(string) == str
        self.string = string

    def save(self):
        return myxml.Tag("string", self.string, [])



class SubInfo(Info):
    @staticmethod
    def Load(tag):
        assert tag.name == "subinfo"
        return SubInfo(tag.get_sub_str("title"), Info.LoadInfos(tag.get_sub_tag("infos")))
    
    def __init__(self, title, infos):
        Info.__init__(self)
        for info in infos:
            assert issubclass(type(info), Info)
        assert type(title) == str
        self.title = title
        self.infos = infos

    def save(self):
        return myxml.Tag("subinfo", "", [myxml.Tag("title", self.title, []),
                                         Info.SaveInfos(self.infos)])

    def get_sub_infos(self, title):
        return [info for info in [info for info in self.infos if type(info) == SubInfo] if info.title == title]
    def get_info_dates(self):
        return [info for info in self.infos if type(info) == Date]
    def get_info_strings(self):
        return [info for info in self.infos if type(info) == String]



class Date(Info):
    @staticmethod
    def Load(tag):
        assert tag.name == "date"
        assert tag.string == ""
        return Date(tag.get_sub_str("day"),
                    tag.get_sub_str("month"),
                    tag.get_sub_str("year"),
                    tag.get_sub_strs("tag"))
    
    def __init__(self, day, month, year, tags):
        Info.__init__(self)
        for x in [day, month, year]:
            assert type(x) == str or x is None
        for tag in tags:
            assert type(tag) == str
        self.day = day
        self.month = month
        self.year = year
        self.tags = tags

    def get_date_string(self):
        tag_str = " ".join(self.tags)
        if self.day is None and self.month is None and not self.year is None:
            return (tag_str + " " + self.year).strip(" ")
        return (tag_str + " " + "-".join(["??" if self.day is None else self.day,
                                          "??" if self.month is None else self.month,
                                          "????" if self.year is None else self.year])).strip(" ")

    def save(self):
        sub_tags = []
        if not self.day is None:
            sub_tags.append(myxml.Tag("day", self.day, []))
        if not self.month is None:
            sub_tags.append(myxml.Tag("month", self.month, []))
        if not self.year is None:
            sub_tags.append(myxml.Tag("year", self.year, []))
        sub_tags.extend([myxml.Tag("tag", tag, []) for tag in self.tags])
        return myxml.Tag("date", "", sub_tags)



class EntityPointer(Info):
    def __init__(self, to_entity):
        Info.__init__(self)
        assert issubclass(type(to_entity), Entity)
        self.from_entity = None
        self.to_entity = to_entity

    def replace_entity(self, old_entity, new_entity):
        assert (t := type(old_entity)) == type(new_entity)
        assert issubclass(t, Entity)
        if self.from_entity == old_entity:
            self.from_entity = new_entity
        if self.to_entity == old_entity:
            self.to_entity = new_entity
    
    def delete(self):
        raise NotImplementedError()



class ParentPointer(EntityPointer):
    @staticmethod
    def Load(tag, entity_lookup):
        return ParentPointer(entity_lookup[tag.get_sub_str("id")])
        
    def __init__(self, parent):
        assert issubclass(type(parent), Person)
        EntityPointer.__init__(self, parent)
        parent.parent_pointers.append(self)

    def delete(self):
        self.from_entity.parents.remove(self)
        self.to_entity.parent_pointers.remove(self)

    def save(self, ident_lookup):
        return myxml.Tag("parent", "", [myxml.Tag("id", ident_lookup[self.to_entity], [])])



class ChildPointer(EntityPointer):
    @staticmethod
    def Load(tag, entity_lookup):
        return ChildPointer(entity_lookup[tag.get_sub_str("id")],
                            {"yes" : True, "no" : False}[tag.get_sub_str("adoption")])
    
    def __init__(self, child, adopted):
        EntityPointer.__init__(self, child)
        assert issubclass(type(child), Person)
        assert type(adopted) == bool
        self.adopted = adopted
        child.child_pointers.append(self)

    def delete(self):
        self.from_entity.children.remove(self)
        self.to_entity.child_pointers.remove(self)

    def save(self, ident_lookup):
        return myxml.Tag("child", "", [myxml.Tag("id", ident_lookup[self.to_entity], []),
                                       myxml.Tag("adoption", {True : "yes", False : "no"}[self.adopted], [])])

        

class ImageEntity(EntityPointer):
    class Rect():
        @staticmethod
        def Load(tag):
            assert tag.name == "rect"
            assert tag.string == ""
            return ImageEntity.Rect([float(tag.get_sub_str("x")),
                                     float(tag.get_sub_str("y")),
                                     float(tag.get_sub_str("w")),
                                     float(tag.get_sub_str("h"))])
        
        def __init__(self, rect):
            assert len(rect) == 4 and all([type(n) == float and 0 <= n <= 1 for n in rect]) and rect[0] + rect[2] <= 1 and rect[1] + rect[3] <= 1
            self.rect = rect

        def save(self):
            return myxml.Tag("rect", "", [myxml.Tag("x", repr(self.rect[0]), []),
                                          myxml.Tag("y", repr(self.rect[1]), []),
                                          myxml.Tag("w", repr(self.rect[2]), []),
                                          myxml.Tag("h", repr(self.rect[3]), [])])
                        
    

    @staticmethod
    def Load(tag, entity_lookup):
        assert tag.name == "image_entity"
        assert tag.string == ""

        def load_rect():
            rect_tags = tag.get_sub_tags("rect")
            if len(rect_tags) == 0:
                return ImageEntity.Rect([0.0, 0.0, 1.0, 1.0])
            elif len(rect_tags) == 1:
                return ImageEntity.Rect.Load(rect_tags[0])
            else:
                raise Exception()
        
        return ImageEntity(entity_lookup[tag.get_sub_str("id")], load_rect(), {"yes" : True, "no" : False}[tag.get_sub_str("usable")])
        
    def __init__(self, entity, rect, usable):
        EntityPointer.__init__(self, entity)
        assert type(rect) == ImageEntity.Rect
        assert type(usable) == bool
        self.rect = rect
        self.usable = usable
        entity.image_pointers.append(self)
        self.surface = None


    def delete(self):
        self.from_entity.image_entities.remove(self)
        self.to_entity.image_pointers.remove(self)


    def save(self, ident_lookup):
        return myxml.Tag("image_entity", "", [myxml.Tag("id", ident_lookup[self.to_entity], []),
                                              myxml.Tag("usable", {True : "yes", False : "no"}[self.usable], []),
                                              self.rect.save()])
        





class Entity(): 
    def __init__(self, ident, infos):
        assert type(ident) == str
        self.ident = ident
        for info in infos:
            assert issubclass(type(info), Info)
        self.infos = infos #list of info objects
        self.image_pointers = []

    def get_sub_infos(self, title):
        return [info for info in [info for info in self.infos if type(info) == SubInfo] if info.title == title]
    def get_info_dates(self):
        return [info for info in self.infos if type(info) == Date]
    def get_info_strings(self):
        return [info for info in self.infos if type(info) == String]

    def delete(self):
        for pointer in self.image_pointers[:]:
            pointer.delete()

    def get_colour(self):
        raise NotImplementedError()


class Person(Entity):
    @staticmethod
    def Load(tag):
        assert tag.name == "person"
        return Person(tag.get_sub_str("id"), Info.LoadInfos(tag.get_sub_tag("infos")))

    @staticmethod
    def merge(p1, p2):
        assert type(p1) == type(p2) == Person
        new_person = Person(p1.infos + p2.infos)
        new_person.parent_pointers = p1.parent_pointers + p2.parent_pointers
        new_person.child_pointers = p1.child_pointers + p2.child_pointers
        new_person.image_pointers = p1.image_pointers + p2.image_pointers
        for pointer in new_person.parent_pointers + new_person.child_pointers + new_person.image_pointers:
            pointer.to_entity = new_person
        return new_person

##    class Relation():
##        class Element():
##            def __init__(self, from_person, to_person, kind, d_gen, frac, sex, info = {}):
##                assert type(from_person) == Person
##                assert type(to_person) == Person
##                assert type(frac) == fractions.Fraction and 0 <= frac <= 1
##                assert kind in ["U", "D", "S", "P"]
##                self.from_person = from_person
##                self.to_person = to_person
##                self.kind = kind
##                self.d_gen = d_gen
##                self.frac = frac
##                self.sex = sex
##                self.info = info
##
##            @property
##            def frac_mult(self):
##                h = fractions.Fraction(1, 2)
##                return {"U" : h, "S" : h, "D" : h, "P" : 0}[self.kind]
##                
##        def __init__(self, elements):
##            for element in elements:
##                assert type(element) == Person.Relation.Element
##            self.elements = elements
##
##        @property
##        def from_person(self):
##            return self.elements[0].from_person
##
##        @property
##        def to_person(self):
##            return self.elements[-1].to_person
##
##        @property
##        def d_gen(self):
##            return sum([e.d_gen for e in self.elements])
##
##        @property
##        def normal_frac_mult(self):
##            t = 1
##            for e in self.elements:
##                t *= e.frac_mult
##            return t
##        @property
##        def actual_frac_mult(self):
##            if len(self.elements) == 0:
##                return 1
##            else:
##                return self.from_person.get_relationship_frac(self.to_person)
##        @property
##        def frac_quotient(self):
##            fracint = lambda x : fractions.Fraction(x, 1) if type(x) != fractions.Fraction else x
##            normal_frac_mult = self.normal_frac_mult
##            if normal_frac_mult == 0:
##                if self.actual_frac_mult != 0:
##                    print("insest lol")
##                return 0
##            else:
##                return self.actual_frac_mult / fracint(self.normal_frac_mult)
##
##        def __add__(self, other):
##            if type(other) == Person.Relation:
##                return Person.Relation(self.elements + other.elements)
##            return NotImplemented
##
##        def get_string(self):
##            #return str(self.normal_frac_mult) + " " + str(self.actual_frac_mult) + " " +  "".join([e.kind for e in self.elements])
##            if len(self.elements) == 0:
##                return "self"
##            else:
##                split_elements = []
##                split_element = []
##                for element in self.elements:
##                    if len(split_element) == 0:
##                        split_element.append(element)
##                    else:
##                        next_kind = element.kind
##                        prev_kind = split_element[-1].kind
##                        if next_kind in {"U" : ["U", "S", "D"], "S" : ["D"], "D" : ["D"], "P" : []}[prev_kind]:
##                            split_element.append(element)
##                        else:
##                            split_elements.append(split_element)
##                            split_element = [element]
##                split_elements.append(split_element)
##
##                def split_element_to_words(elements):
##                    def split_element_to_name(elements):
##                        assert len(elements) != 0
##                        #partners                   P
##                        #siblings                   S
##                        if len(elements) == 1:
##                            e = elements[0]
##                            if e.kind == "P":
##                                return {"male" : "husband", "female" : "wife"}.get(e.sex, "spouse") if e.info.get("married", False) else {"male" : "boif", "female" : "gurlf"}.get(e.sex, "partner")
##                            if e.kind == "S":
##                                return {"male" : "brother", "female" : "sister"}.get(e.sex, "sibling")
##                        assert not "P" in [e.kind for e in elements]
##                        #ancestors                  U...U
##                        if all([e.kind == "U" for e in elements]):
##                            name = {"male" : "father", "female" : "mother"}.get(elements[-1].sex, "parent")
##                            count = len(elements) - 1
##                            if count > 0:
##                                name = "grand" + name
##                                count -= 1
##                            if count > 0:
##                                name = numwords.repeat_word(count, "great") + " " + name
##                            return name
##                        #decendents                 D...D
##                        if all([e.kind == "D" for e in elements]):
##                            name = {"male" : "son", "female" : "daughter"}.get(elements[-1].sex, "child")
##                            count = len(elements) - 1
##                            if count > 0:
##                                name = "grand" + name
##                                count -= 1
##                            if count > 0:
##                                name = numwords.repeat_word(count, "great") + " " + name
##                            return name
##                        #aunts & uncles             U...US
##                        if elements[-1].kind == "S":
##                            name = {"male" : "uncle", "female" : "aunt"}.get(elements[-1].sex, "auncle")
##                            count = len(elements) - 2
##                            if count > 0:
##                                name = "grand" + name
##                                count -= 1
##                            if count > 0:
##                                name = numwords.repeat_word(count, "great") + " " + name
##                            return name
##                        #nephews & necsess          SD...D
##                        if elements[0].kind == "S":
##                            name = {"male" : "nephew", "female" : "niece"}.get(elements[-1].sex, "nibling")
##                            count = len(elements) - 2
##                            if count > 0:
##                                name = "grand" + name
##                                count -= 1
##                            if count > 0:
##                                name = numwords.repeat_word(count, "great") + " " + name
##                            return name
##                        #cousins                    U...USD...D
##                        kinds = [e.kind for e in elements]
##                        u, d = kinds.count("U"), kinds.count("D")
##                        c, r = min(u, d), abs(u - d)
##                        if c == 1 and r == 0:
##                            return "cousin"
##                        elif r == 0:
##                            return numwords.ordinal(c) + " cousin"
##                        elif c == 1:
##                            return "cousin " + numwords.times(r) + " removed"
##                        else:
##                            return numwords.ordinal(c) + " cousin " + numwords.times(r) + " removed"
##                    name = split_element_to_name(elements)
##                    rel = Person.Relation(elements)
##                    if rel.normal_frac_mult == 0 and rel.actual_frac_mult != 0:
##                        return numwords.fraction(rel.actual_frac_mult) + " insest " + name
##                    else:
##                        frac = rel.frac_quotient
##                        if frac == 1 or any(e.kind == "P" for e in elements):
##                            return name
##                        elif frac == 0:
##                            return "non-blood " + name
##                        else:
##                            return numwords.multiplier(frac) + " " + name
##
##                return "'s ".join([split_element_to_words(split_element) for split_element in split_elements]).replace(" ", "\n")
                                        
    
    def __init__(self, ident, infos):
        Entity.__init__(self, ident, infos)
        self.parent_pointers = [] #pointers of partnerships for which this person is a parent
        self.child_pointers = [] #pointers of partnerships for which this person is a child

##        self.relationship_cache = {} # {person : Person.Relation}
##        self.relationship_to_check = set([])
##        self.relationship_frac = None # {person : fractions.Fraction} for frac of genes shared
##        self.reset_relationship_cache()

    def __str__(self):
        return "Person(" + self.name() + ")"
    def name(self):
        return " ".join(itertools.chain(self.get_first_names(), ((n[0] if len(n) >= 1 else "") for n in self.get_last_names())))

    def delete(self):
        super().delete()
        for pointer in self.parent_pointers[:]:
            pointer.delete()
        for pointer in self.child_pointers[:]:
            pointer.delete()


##    def reset_relationship_cache(self):
##        self.relationship_cache = {self : Person.Relation([])}
##        self.relationship_to_check = set([self])
##        self.relationship_frac = None
##
##    def get_relationship_frac(self, person):
##        if self.relationship_frac is None:
##            self.relationship_frac = {self : 1}
##            locked = set(self.relationship_frac.keys())
##            to_check = set(self.relationship_frac.keys())
##            while len(to_check) != 0:
##                new_to_check = set({})
##                for p1 in to_check:
##                    for p2 in p1.get_parents():
##                        if not p2 in locked:
##                            if not p2 in self.relationship_frac:
##                                self.relationship_frac[p2] = 0
##                            self.relationship_frac[p2] += fractions.Fraction(1, 2) * self.relationship_frac[p1]
##                            new_to_check.add(p2)
##                to_check = new_to_check
##
##            locked = set(self.relationship_frac.keys())
##            to_check = set(self.relationship_frac.keys())
##            while len(to_check) != 0:
##                new_to_check = set({})
##                for p1 in to_check:
##                    for p2 in p1.get_siblings():
##                        if not p2 in locked:
##                            if not p2 in self.relationship_frac:
##                                self.relationship_frac[p2] = 0
##
##                            p1pairs = set(p1.get_parents())
##                            p2pairs = set(p2.get_parents())
##                            p1p2pairs = p1pairs & p2pairs
##
##                            p1p2pairs_len = len(p1p2pairs)
##                            p1pairs_len = len(p1pairs)
##                            p2pairs_len = len(p2pairs)
##                            while p1pairs_len < 2 and p2pairs_len < 2:
##                                p1pairs_len += 1
##                                p2pairs_len += 1
##                                p1p2pairs_len += 1
##                            while p1pairs_len < 2:
##                                p1pairs_len += 1
##                            while p2pairs_len < 2:
##                                p2pairs_len += 1
##                                
##                                
##                            self.relationship_frac[p2] += fractions.Fraction(p1p2pairs_len, p1pairs_len * p2pairs_len) * self.relationship_frac[p1]
##                            new_to_check.add(p2)
##                            locked.add(p2)
##                to_check = new_to_check
##                
##            locked = set(self.relationship_frac.keys())
##            to_check = set(self.relationship_frac.keys())
##            while len(to_check) != 0:
##                new_to_check = set({})
##                for p1 in to_check:
##                    for p2 in p1.get_children():
##                        if not p2 in locked:
##                            if not p2 in self.relationship_frac:
##                                self.relationship_frac[p2] = 0
##                            self.relationship_frac[p2] += fractions.Fraction(1, 2) * self.relationship_frac[p1]
##                            new_to_check.add(p2)
##                to_check = new_to_check  
##        return self.relationship_frac.get(person, 0)
##
##    def get_relationship_words(self, person):
##        rel = self.get_relationship(person)
##        if not rel is None:
##            return rel.get_string()
##        else:
##            return "unrelated"
##
##    def get_relationship(self, person):
##        assert type(person) == Person
##        while not person in self.relationship_cache and len(self.relationship_to_check) != 0:
##            new_to_check = set([])
##            for p1 in self.relationship_to_check:
##                done = False
##                while not done:
##                    done = True
##                    for p2 in p1.get_parents():
##                        if not p2 in self.relationship_cache:
##                            self.relationship_cache[p2] = self.relationship_cache[p1] + Person.Relation([Person.Relation.Element(p1, p2, "U", -1, fractions.Fraction(1, 2), p2.get_sex())])
##                            new_to_check.add(p2)
##                            done = False
##                for p2 in p1.get_siblings():
##                    if not p2 in self.relationship_cache:
##                        self.relationship_cache[p2] = self.relationship_cache[p1] + Person.Relation([Person.Relation.Element(p1, p2, "S", 0, fractions.Fraction(1, 2), p2.get_sex())])
##                        new_to_check.add(p2)
##                done = False
##                while not done:
##                    done = True
##                    for p2 in p1.get_children():
##                        if not p2 in self.relationship_cache:
##                            self.relationship_cache[p2] = self.relationship_cache[p1] + Person.Relation([Person.Relation.Element(p1, p2, "D", 1, fractions.Fraction(1, 2), p2.get_sex())])
##                            new_to_check.add(p2)
##                            done = False
##                for part in [pointer.from_entity for pointer in p1.parent_pointers]:
##                    for p2 in [pointer.to_entity for pointer in part.parents]:
##                        if not p2 in self.relationship_cache:
##                            self.relationship_cache[p2] = self.relationship_cache[p1] + Person.Relation([Person.Relation.Element(p1, p2, "P", 0, fractions.Fraction(0, 1), p2.get_sex(), {"married" : part.get_married()})])
##                            new_to_check.add(p2)
##            self.relationship_to_check = new_to_check
##        if person in self.relationship_cache:
##            return self.relationship_cache[person]
##        else:
##            return None


    def get_parents(self):
        for pointer in self.child_pointers:
            part = pointer.from_entity
            for person_pointer in part.parents:
                yield person_pointer.to_entity

    def get_children(self):
        for pointer in self.parent_pointers:
            part = pointer.from_entity
            for person_pointer in part.children:
                yield person_pointer.to_entity

    def get_siblings(self):
        sibligs = set([])
        for parent in self.get_parents():
            for child in parent.get_children():
                sibligs.add(child)
        for pointer in self.child_pointers:
            part = pointer.from_entity
            for person_pointer in part.children:
                sibligs.add(person_pointer.to_entity)
        return sibligs

    def get_partners(self):
        for pointer in self.parent_pointers:
            part = pointer.from_entity
            for person_pointer in part.parents:
                yield person_pointer.to_entity

    def get_parent_parts(self):
        for pointer in self.parent_pointers:
            yield pointer.from_entity


    def get_child_parts(self):
        for pointer in self.child_pointers:
            yield pointer.from_entity


    def get_decedents(self, found = None):
        if found is None:
            found = set([])
        if not self in found:
            found |= set([self])
        for person in self.get_children():
            if not person in found:
                found |= person.get_decedents(found)
        return found

    def get_ancestors(self, found = None):
        if found is None:
            found = set([])
        if not self in found:
            found |= set([self])
        for person in self.get_parents():
            if not person in found:
                found |= person.get_ancestors(found)
        return found
        

    def get_name_infos(self):
        return sum([info.infos for info in self.get_sub_infos("name")], [])
    def get_first_names(self):
        return [info.string for info in sum([info.get_info_strings() for info in [info for info in self.get_name_infos() if type(info) == SubInfo] if info.title == "first"], []) if info.string != ""]
    def get_middle_names(self):
        return [info.string for info in sum([info.get_info_strings() for info in [info for info in self.get_name_infos() if type(info) == SubInfo] if info.title == "middle"], []) if info.string != ""]
    def get_last_names(self):
        return [info.string for info in sum([info.get_info_strings() for info in [info for info in self.get_name_infos() if type(info) == SubInfo] if info.title == "last"], []) if info.string != ""]
    def get_known_as_names(self):
        return [info.string for info in sum([info.get_info_strings() for info in [info for info in self.get_name_infos() if type(info) == SubInfo] if info.title == "known as"], []) if info.string != ""]

    def get_sex(self):
        sex_infos = self.get_sub_infos("sex")
        if len(sex_infos) != 1:
            return None
        else:
            sexes = [info.string for info in sex_infos[0].get_info_strings()]
            if len(sexes) != 1:
                return None
            else:
                return sexes[0]

    def get_colour(self):
        sex = self.get_sex()
        if sex is None:
            return [128, 128, 128]
        elif sex == "male":
            return [64, 160, 255]
        elif sex == "female":
            return [255, 64, 255]
        else:
            return [112, 64, 255]
            

    def save(self, ident_lookup):
        return myxml.Tag("person", "", [myxml.Tag("id", ident_lookup[self], []),
                                        Info.SaveInfos(self.infos)])



class Partnership(Entity):
    @staticmethod
    def Load(tag, entity_lookup):
        assert tag.name == "partnership"
        return Partnership(tag.get_sub_str("id"),
                           Info.LoadInfos(tag.get_sub_tag("infos")),
                           [ParentPointer.Load(sub_tag, entity_lookup) for sub_tag in tag.get_sub_tag("parents").get_sub_tags("parent")],
                           [ChildPointer.Load(sub_tag, entity_lookup) for sub_tag in tag.get_sub_tag("children").get_sub_tags("child")])

    @staticmethod
    def merge(p1, p2):
        assert type(p1) == type(p2) == Partnership
        new_part = Partnership(p1.infos + p2.infos, p1.parents + p2.parents, p1.children + p2.children)
        new_part.image_pointers = p1.image_pointers + p2.image_pointers
        for pointer in new_part.image_pointers:
            pointer.to_entity = new_part
        return new_part
    
    def __init__(self, ident, infos, parents, children):
        Entity.__init__(self, ident, infos)
        for parent in parents:
            assert issubclass(type(parent), ParentPointer)
        for child in children:
            assert issubclass(type(child), ChildPointer)
        self.parents = parents
        self.children = children
        for parent_pointer in self.parents:
            parent_pointer.from_entity = self
        for child_pointer in self.children:
            child_pointer.from_entity = self

    def __str__(self):
        return "Partnership(" + ", ".join(parent.to_entity.name() for parent in self.parents) + " -> " + ", ".join(child.to_entity.name() for child in self.children) + ")"

    def delete(self):
        super().delete()
        for pointer in self.parents + self.children:
            pointer.delete()

    def get_married(self):
        return len([info for info in [info for info in self.infos if type(info) == SubInfo] if info.title == "marriage"]) >= 1
    def get_divorced(self):
        return len([info for info in [info for info in self.infos if type(info) == SubInfo] if info.title == "divorce"]) >= 1

    def is_child(self, person):
        assert issubclass(type(person), Person)
        return any([child_pointer.to_entity == person for child_pointer in self.children])

    def is_parent(self, person):
        assert issubclass(type(person), Person)
        return any([parent_pointer.to_entity == person for parent_pointer in self.parents])

    def add_child(self, person):
        assert not self.is_child(person)
        child_pointer = ChildPointer(person, False)
        child_pointer.from_entity = self
        self.children.append(child_pointer)

    def add_parent(self, person):
        assert not self.is_parent(person)
        parent_pointer = ParentPointer(person)
        parent_pointer.from_entity = self
        self.parents.append(parent_pointer)

    def remove_child(self, person):
        assert self.is_child(person)
        for child_pointer in self.children[:]:
            if child_pointer.to_entity == person:
                child_pointer.delete()

    def remove_parent(self, person):
        assert self.is_parent(person)
        for parent_pointer in self.parents[:]:
            if parent_pointer.to_entity == person:
                parent_pointer.delete()
                

    def save(self, ident_lookup):
        return myxml.Tag("partnership", "", [myxml.Tag("id", ident_lookup[self], []),
                                             Info.SaveInfos(self.infos),
                                             myxml.Tag("parents", "", [parent.save(ident_lookup) for parent in self.parents if parent.to_entity in ident_lookup]),
                                             myxml.Tag("children", "", [child.save(ident_lookup) for child in self.children if child.to_entity in ident_lookup])])


class Image(Entity):
    @staticmethod
    def Load(tag, entity_lookup):
        assert tag.name == "image"
        return Image(tag.get_sub_str("id"), Info.LoadInfos(tag.get_sub_tag("infos")), tag.get_sub_str("file"), [ImageEntity.Load(sub_tag, entity_lookup) for sub_tag in tag.get_sub_tag("image_entities").get_sub_tags("image_entity")])
    
    def __init__(self, ident, infos, path, image_entities):
        Entity.__init__(self, ident, infos)
        assert type(path) == str
        for image_entity in image_entities:
            assert issubclass(type(image_entity), ImageEntity)
        self.path = path
        self.image_entities = image_entities
        for image_entity in image_entities:
            image_entity.from_entity = self
            
        self.surface = None

    def __str__(self):
        return "Image(" + self.path + ")"

    def is_pictured(self, entity):
        assert issubclass(type(entity), Entity)
        return any([pointer.to_entity == entity for pointer in self.image_entities])

    def add_pictured(self, entity):
        assert not self.is_pictured(entity)
        image_pointer = ImageEntity(entity, ImageEntity.Rect([0.0, 0.0, 1.0, 1.0]), False)
        image_pointer.from_entity = self
        self.image_entities.append(image_pointer)

    def remove_pictured(self, entity):
        assert self.is_pictured(entity)
        for pointer in self.image_entities[:]:
            if pointer.to_entity == entity:
                pointer.delete()


    def save(self, ident_lookup):
        return myxml.Tag("image", "", [myxml.Tag("id", ident_lookup[self], []),
                                       myxml.Tag("file", self.path, []),
                                       Info.SaveInfos(self.infos),
                                       myxml.Tag("image_entities", "", [image_entity.save(ident_lookup) for image_entity in self.image_entities if image_entity.to_entity in ident_lookup])])




class Tree():
    @staticmethod
    def Load(tag):
        assert tag.name == "tree"
        entity_lookup = {}
        for sub_tag in tag.get_sub_tags("person"):
            entitiy = Person.Load(sub_tag)
            assert not entitiy.ident in entity_lookup
            entity_lookup[entitiy.ident] = entitiy
        for sub_tag in tag.get_sub_tags("partnership"):
            entitiy = Partnership.Load(sub_tag, entity_lookup)
            assert not entitiy.ident in entity_lookup
            entity_lookup[entitiy.ident] = entitiy
        for sub_tag in tag.get_sub_tags("image"):
            entitiy = Image.Load(sub_tag, entity_lookup)
            assert not entitiy.ident in entity_lookup
            entity_lookup[entitiy.ident] = entitiy
        return Tree(entity_lookup)
    
    def __init__(self, entity_lookup = {}):
        for ident, entity in entity_lookup.items():
            assert type(ident) == str
            assert issubclass(type(entity), Entity)
        self.entity_lookup = entity_lookup

    def new_ident(self):
        for x in itertools.count():
            if not str(x) in self.entity_lookup:
                return x

    @property
    def people(self):
        return tuple(entity for entity in self.entity_lookup.values() if type(entity) == Person)

##    def structure_changed(self):
##        for entity in self.entities:
##            if type(entity) == Person:
##                entity.reset_relationship_cache()


##    def add_entity(self, entity):
##        assert issubclass(type(entity), Entity)
##        assert not entity in self.entity_lookup.values()
##        self.entity_lookup[self.new_ident] = entity
##
##    def remove_entity(self, entity):
##        assert issubclass(type(entity), Entity)
##        entity.delete()
##        self.entities.remove(entity)
        

##    def replace_entity(self, old_entity, new_entity):
##        assert type(old_entity) == type(new_entity)
##        assert issubclass(type(old_entity), Entity)
##        assert issubclass(type(new_entity), Entity)
##        assert old_entity in self.entities
##        assert not new_entity in self.entities
##        self.entities.remove(old_entity)
##        for entity in self.entities:
##            entity.replace_entity(old_entity, new_entity)
##        self.entities.append(new_entity)


##    def merge_entities(self, entities, merger):
##        assert merger in [Person.merge, Partnership.merge]
##        if len(entities) >= 2:
##            t = type(entities[0])
##            assert all(type(entity) == t for entity in entities)
##            assert issubclass(t, Entity)
##
##            merge_entity = merger(entities[0], entities[1])
##            for i in range(2, len(entities)):
##                merge_entity = merger(merge_entity, entities[i])
##            assert type(merge_entity) == t
##
##            for entity in entities:
##                self.entities.remove(entity)
##            self.entities.append(merge_entity)
##        #Tree.Load(self.save())

    def save(self):
        return myxml.Tag("tree", "", [entity.save(self.entity_lookup) for entity in self.entities])

    def sprint(self):
        print("TREE", self)
        for ident, entity in self.entity_lookup.items():
            print(ident, entity)

    def digraph(self):
        #graph whose verticies are either people or partnerships
        #people connect to partnerships
        #partnerships connect to people
        #edges are directed towards decendents
        #there should be no cycles (otherwise timetravel happened)

        G = nx.DiGraph()
        for ident, entity in self.entity_lookup.items():
            if type(entity) in [Person, Partnership]:
                G.add_node(ident)

        for ident, entity in self.entity_lookup.items():
            if type(entity) == Partnership:
                for parent_pointer in entity.parents:
                    G.add_edge(parent_pointer.to_entity.ident, ident)
                for child_pointer in entity.children:
                    G.add_edge(ident, child_pointer.to_entity.ident)

        assert nx.is_directed_acyclic_graph(G) #assert no time travel
        return G

##    def planar_centered_graph(self, entity):
##        G = self.digraph()
##        assert entity.ident in G.nodes()
##        print(G, entity)
##
##        #ancestors
##        #descendants
##
##        print(nx.ancestors(G, entity.ident))
##        print(nx.descendants(G, entity.ident))
##
##        return 

        
        
        

        

        

        
        




















    
