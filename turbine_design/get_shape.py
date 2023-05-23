from .turbigen import three_dimensional_stage, ohmesh 
import pkg_resources

def get_shape(params):
    """Function to return shape from a parameters object.

    Returns
    -------
    nb: (nrow,) array
        Numbers of blades.
    h: (npts, 2) array
        x-r hub coordinates.
    c: (npts, 2) array
        x-r casing coordinates.
    ps: nested list ps[row][section][point on section, x/r/t]
        Pressure-side aerofoil coordinates.
    ss:
        Same as `ps` for the suction side.

    """

    # The meshing routines need "Section generators", callables that take a
    # span fraction and output a blade section
    def vane_section(spf):
        return params.interpolate_section(0, spf)

    def blade_section(spf):
        return params.interpolate_section(1, spf, is_rotor=True)

    sect_generators = [vane_section, blade_section]

    # For brevity
    Dstg = params.dimensional_stage

    # Evaluate shape
    _, _, nb, ps, ss, h, c, _ = ohmesh.get_sections_and_annulus_lines(
        params.dx_c, Dstg.rm, Dstg.Dr, Dstg.cx, Dstg.s, params.tau_c, sect_generators
        )

    return nb, h, c, ps, ss

def get_coordinates():
    
    package_path = pkg_resources.resource_filename('turbine_design','')
    json_subfolder_path = '/'.join((package_path, 'turbine_json'))

    # Load a parameter set
    params = three_dimensional_stage.StageParameterSet.from_json(f"{json_subfolder_path}/turbine_params.json")

    # Extract shape
    nb, h, c, ps, ss = get_shape(params)
    
    return nb, h, c, ps, ss
