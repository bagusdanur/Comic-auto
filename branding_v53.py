"""V5.3 preserves V5.2 source cleaning and uses larger, locally checked logos."""
from branding_v52 import apply_branding_v52
from logo_layout_v53 import fit_larger_logo


def apply_branding_v53(*args,**kwargs):
    return apply_branding_v52(*args,**kwargs,logo_layout=fit_larger_logo,branding_version='v5.3')
