from enum import Enum
class DataType(Enum):
    FLOAT = float
    INT = int
    BOOL = bool
    STR = str

class AttributeCollector:
    def __init__(self, study) -> None:
        self.study = study
        self._attributes = {}

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes

    def check_data_type(self, name, value, parent):
        # checks if the data type entered is acceptable
        attribute_type = self._attributes[name][0]
        if value is None and parent is not None:
            self._attributes[name][1] = value
            return True
        elif attribute_type == DataType.FLOAT and type(value) == int and value >= 0:
            self._attributes[name][1] = value
            return True
        elif attribute_type.value == type(value) and value >= 0:
            self._attributes[name][1] = value
            return True
        else:
            print("Invalid value! Please input a positive " + str(attribute_type.value.__name__))
            return False

    def load_attributes_collector(self, attributes_dict):
        for key, value in attributes_dict.items():
            for datatype in DataType:
                if datatype.name == value[0]:
                    attributes_dict[key][0] = datatype
        self._attributes = attributes_dict

    def save_attributes_collector(self):
        attributes_dict = {}
        for name, values in self._attributes.items():
            attributes_dict[name] = [values[0].name, values[1]]
        return attributes_dict

class Component:
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        self._name = name
        self.study = study
        self._mode_name = mode_name
        if not new_mode:
            self._modes = []
            self._parent = None
            if parent_attributes is None:
                parent_attributes = AttributeCollector(self.study)
                self._attributes_collector = parent_attributes
                if hasattr(self, "setup"):
                    self.setup()
            else:
                self._attributes_collector = parent_attributes
        else:
            self._attributes_collector = AttributeCollector(self.study)

    @property
    def name(self):
        return self._name

    @property
    def mode_name(self):
        return self._mode_name

    @mode_name.setter
    def mode_name(self, name):
        self._mode_name = name

    @property
    def modes(self):
        return self._modes

    @property
    def parent(self):
        return self._parent

    @property
    def attributes_collector(self):
        return self._attributes_collector

    @attributes_collector.setter
    def attributes_collector(self, attributes_collector):
        self._attributes_collector = attributes_collector

    def get_mode(self, name: str):
        if name == self._mode_name:
            return self
        for mode in self._modes:
            if mode.mode_name == name:
                return mode

    def get_mode_names(self):
        mode_names = [self._mode_name]
        for mode in self._modes:
            mode_names.append(mode.mode_name)
        return mode_names

    def create_attribute(self, name, attribute_type:DataType, default_value):
        self._attributes_collector.attributes[name] = [attribute_type, default_value]

    def get_attribute(self, name:str):
        if name in self._attributes_collector.attributes:
            return self._attributes_collector.attributes[name][1]
        elif self._parent is not None and name in self._parent.attributes_collector.attributes:
            return self._parent.attributes_collector.attributes[name][1]
        else:
            return None

    def set_attribute(self, name, value):
        #TODO: add some logic if some variables are linked to other variables (aspect ratio, chord, span, etc.)
        if name in self._attributes_collector.attributes:

            change = self._attributes_collector.check_data_type(name=name, value=value, parent=self._parent)

        elif name in self._parent.attributes_collector.attributes:
            attribute_type = self._parent.attributes_collector.attributes[name][0]
            parent_value = self._parent.get_attribute(name)
            self.create_attribute(name, attribute_type=attribute_type, default_value=parent_value)
            change = self._attributes_collector.check_data_type(name=name, value=value, parent=self._parent)
            if not change:
                self._attributes_collector.attributes.pop(name)
        else:
            change = False

        keys = ['span', 'aspect_ratio', 'area', 'taper', 'volume', 'thickness', 'root_chord'] # list of quantites with relationships
        if name in keys and change:
            self.link_components(name=name, keys=keys)

    def link_components(self, name, keys):

        values = []
        for item in keys:
            values.append(self.get_attribute(item))

        span,aspect_ratio,area,taper,volume,thickness,root_chord = values # unpack for simplicity when writing equations

        relations_dict = {"span":[area,aspect_ratio, taper, root_chord],
                 "taper":[area,aspect_ratio, span, root_chord],
                 "root_chord": [area,aspect_ratio,span,taper],
                 'area': [span,aspect_ratio,taper,root_chord],
                 'aspect_ratio': [span,area, taper, root_chord],
                 'thickness':[volume,area],
                 'volume':[thickness,area]}

        params = relations_dict[name]


        # make sure change is tracked only when changing attribute, not during initial defintition
        if 0 not in params:

            # change area and aspect ratio based on change on span, taper, pr root chord
            if name =='span' or name == 'taper' or name == 'root_chord':
                self._attributes_collector.attributes['area'][1] = span/2 *(1+taper)*root_chord
                self._attributes_collector.attributes['aspect_ratio'][1] = span**2/self.get_attribute('area')

            # change span and aspect ratio if area changes
            elif name == 'area':
                self._attributes_collector.attributes['span'][1] = area*2 / (1+taper) / root_chord
                self._attributes_collector.attributes['aspect_ratio'][1] = self.get_attribute('span')**2/area

            # change area and span if aspect ratio changes
            elif name == 'aspect_ratio':
                self._attributes_collector.attributes['span'][1] = aspect_ratio*(1+taper)*root_chord/2
                self._attributes_collector.attributes['area'][1] = self.get_attribute('span')**2/aspect_ratio

            # change volume if thickness changes
            elif name == 'thickness':
                self._attributes_collector.attributes['volume'][1] = area*thickness

            # change thickness if volume changes
            elif name == 'volume':
                self._attributes_collector.attributes['thickness'][1] = volume/area

    def new_mode(self, mode_name):
        mode_names = self.get_mode_names()
        if mode_name not in mode_names:
            mode = ComponentMode(study=self.study, parent=self, name=self._name, mode_name=mode_name)
            self._modes.append(mode)
            return mode
        else:
            return self.get_mode(mode_name)

    def save_component(self):
        component_dict = {}
        component_dict["mode_name"] = self._mode_name
        component_dict["type"] = type(self).__name__
        component_dict["attributes_collector"] = self._attributes_collector.save_attributes_collector()
        modes_dict = {}
        for i in range(0, len(self._modes)):
            mode = self._modes[i]
            attributes_collector_dict = {}
            for name, values in mode.attributes_collector.attributes.items():
                attributes_collector_dict[name] = [values[0].name, values[1]]
            modes_dict[mode.mode_name] = attributes_collector_dict
        component_dict["modes"] = modes_dict
        return component_dict

class ComponentMode(Component):
    def __init__(self, study, parent, name, mode_name) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, new_mode=True)
        self._parent = parent

class LiftingSurface(Component):

    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            # geometry - can be wing or tail
            self.create_attribute("span", DataType.FLOAT, 0)
            self.create_attribute("root_chord", DataType.FLOAT, 0)
            # self.create_attribute("nPanel", DataType.INT, 0) # number of pannels on lifting surface
            self.create_attribute("area", DataType.FLOAT, 0)
            self.create_attribute("aspect_ratio", DataType.FLOAT, 0)
            self.create_attribute("taper", DataType.FLOAT, 0)
            self.create_attribute("twist", DataType.FLOAT, 0)
            self.create_attribute("dihedral", DataType.FLOAT, 0)
            self.create_attribute("sweep", DataType.FLOAT, 0)
            self.create_attribute("volume", DataType.FLOAT, 0)
            self.create_attribute("thickness", DataType.FLOAT, 0) # thickness to chord ratio

            self.create_attribute("nrotor", DataType.FLOAT, 0) # number of rotors on wing
            self.create_attribute("type_rotor", DataType.INT, 0) # rotor numbers on wing (change to ist or np array?)

            # locations of aerodyanmic center
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

class Fuselage(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            self.create_attribute("length", DataType.FLOAT, 0)
            self.create_attribute("width", DataType.FLOAT, 0)
            self.create_attribute("height", DataType.FLOAT, 0)
            self.create_attribute("wetted_area", DataType.FLOAT, 0)
            self.create_attribute("projected_area", DataType.FLOAT, 0)
            self.create_attribute("projected_area", DataType.FLOAT, 0) # fuselage SL location relative nose - (fraction fuselage length)
            self.create_attribute("aft_length", DataType.FLOAT, 0)
            self.create_attribute("nose_length", DataType.FLOAT, 0)

            # location
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

class Propulsion(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            # geometry
            # self.create_attribute("nengines", DataType.INT, 0)
            self.create_attribute("length", DataType.FLOAT, 0) # drive shaft length

            #  weight
            self.create_attribute("length_multicopter", DataType.FLOAT, 0) # arm length as a fraction of radius
            self.create_attribute("length_multicopter", DataType.FLOAT, 0) # arm length as a fraction of radius
            self.create_attribute("length_multicopter", DataType.FLOAT, 0) # arm length as a fraction of radius

class LandingGear(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            # geometry
            # self.create_attribute("nlanding_gear", DataType.INT, 0)

            # location
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

            # weight
            self.create_attribute('wlg', DataType.FLOAT, 0) # weight landing gear (fraction of max take off weight in NDARC)

class Rotors(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            # geometry
            self.create_attribute("radius", DataType.FLOAT, 0)
            self.create_attribute("chord", DataType.FLOAT, 0)
            self.create_attribute("twist", DataType.FLOAT, 0)
            self.create_attribute("taper_ratio", DataType.FLOAT, 0)
            self.create_attribute("thickness", DataType.FLOAT, 0) #thickness to chord ratio
            self.create_attribute("nblade", DataType.INT, 0)
            # self.create_attribute("nrotor", DataType.INT, 0)
            self.create_attribute("incid_hub", DataType.INT, 0) # orientation of the rotor shaft incidence angle
            self.create_attribute("cant_hub", DataType.INT, 0) # orientation of the rotor shaft cant angle
            self.create_attribute("direction_rotation", DataType.INT, 1) # 1ccw, -1 cw
            self.create_attribute("angle_multicopter", DataType.FLOAT, 0) # angle clockwise from forard
            self.create_attribute("length_multicopter", DataType.FLOAT, 0) # arm length as a fraction of radius
            self.create_attribute("tilt", DataType.FLOAT, 0) # nacelle tilt

            # location
            self.create_attribute("loc_x", DataType.INT, 0) # x coordinate hub
            self.create_attribute("loc_y", DataType.INT, 0) # y coordinate hub
            self.create_attribute("loc_z", DataType.INT, 0) # z coordinate hub

            # weight

class Engines(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            self.create_attribute("wetted_area", DataType.FLOAT, 0)

            # location
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

class Aircraft(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            self.create_attribute("nwing", DataType.INT, 0) # number of wings
            self.create_attribute("nrotor", DataType.INT, 0) # number of rotors
            self.create_attribute("ntail", DataType.INT, 0) # number of tails
            self.create_attribute("nprop", DataType.INT, 0) # number of propulsion groups
            self.create_attribute("nengine_group", DataType.INT, 0) # number of engine groups
            self.create_attribute("nengine_model", DataType.INT, 0) # number of engine models
            self.create_attribute("ncontrol", DataType.INT, 0) # number of controls systems
            self.create_attribute("ntank", DataType.INT, 0) # number of fuel tank systems
            self.create_attribute("nmotor_model", DataType.INT, 0) # number of motor models
            self.create_attribute("ncontrol", DataType.INT, 0) # number of controls systems

            # cg location
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

            # weights
            self.create_attribute('dgw', DataType.FLOAT, 0) # design gross weight
            self.create_attribute('sdgw', DataType.FLOAT, 0) # structural design gross weight
            self.create_attribute('wmto', DataType.FLOAT, 0) # max takeoff weight
            self.create_attribute('kx', DataType.FLOAT, 0) # moment of inertia based on design gross weight
            self.create_attribute('ky', DataType.FLOAT, 0) # moment of inertia based on design gross weight
            self.create_attribute('kz', DataType.FLOAT, 0) # moment of inertia based on design gross weight
            self.create_attribute('npass', DataType.INT, 0) # number of passangers
            self.create_attribute('wpass', DataType.FLOAT, 0) # weight per passanger
            self.create_attribute('wcrew', DataType.FLOAT, 0) # weight of crew
            self.create_attribute('wtrap', DataType.FLOAT, 0) # weight of trapped fluids and oil
            self.create_attribute('wcont', DataType.FLOAT, 0) # contingency weight
            self.create_attribute('wauto_fc', DataType.FLOAT, 0) # weight automated flight control
            self.create_attribute('wrot_fc', DataType.FLOAT, 0) # weight rotary wing flight control
            self.create_attribute('winstrument', DataType.FLOAT, 0) # weight systems and equipment
            self.create_attribute('wavionics', DataType.FLOAT, 0) # weight avionics
            self.create_attribute('welectrical', DataType.FLOAT, 0) # weight electrical systems
            self.create_attribute('wfurnish', DataType.FLOAT, 0) # weight furnishing
            self.create_attribute('wice', DataType.FLOAT, 0) # weight anti icing

class Tanks(Component):
    def __init__(self, study, name, mode_name="baseline", parent_attributes=None, new_mode=False) -> None:
        super().__init__(study=study, name=name, mode_name=mode_name, parent_attributes=parent_attributes, new_mode=new_mode)
        if parent_attributes is None:
            # locations
            self.create_attribute("loc_x", DataType.FLOAT, 0)
            self.create_attribute("loc_y", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)

            # weight
            # add plumbing weight factors?
            self.create_attribute("wfuel_cap", DataType.FLOAT, 0) # fuel capacity
            self.create_attribute("wfuel_density", DataType.FLOAT, 0) # fuel density
            self.create_attribute("loc_z", DataType.FLOAT, 0)
            self.create_attribute("loc_z", DataType.FLOAT, 0)