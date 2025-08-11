build_links = {
    "OpenvspPlugin": {
        "LiftingSurface": {
            "span": ".WING.WingGeom.TotalSpan",
            "root_chord": ".WING.WingGeom.TotalChord"
        }
    }

}

def auto_build_links(solution, vehicle_name, vehicle_mode_name, component):
    for step_name, step_info in solution.steps.items():
        if step_info[0] == "OpenvspPlugin":
            openvsp_build_links = build_links[step_info[0]]
            if type(component).__name__ in openvsp_build_links:
                type_build_links = openvsp_build_links[type(component).__name__]
                for attribute in component.attributes_collector.attributes:
                    if attribute in type_build_links:
                        left = "Vehicles." + vehicle_name + "." + component.name + "." + component.mode_name + "." + attribute
                        right = step_name + "." + component.name + type_build_links[attribute]
                        solution.add_link(left, right)
            elif type(component.parent).__name__ in openvsp_build_links:
                type_build_links = openvsp_build_links[type(component.parent).__name__]
                for attribute in component.attributes_collector.attributes:
                    if attribute in type_build_links:
                        left = "Vehicles." + vehicle_name + "." + vehicle_mode_name + "." + component.name + "." + attribute
                        right = step_name + "." + component.name + type_build_links[attribute]
                        solution.add_link(left, right)