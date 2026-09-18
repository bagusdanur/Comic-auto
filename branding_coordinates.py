"""Map RAW-coordinate regions into a cropped branded preview."""
def project_box(box,report):
    if report.get('branding_engine') not in ('v5.2','v5.3'):return dict(box)
    top=report.get('source_header_trimmed_at') or 0
    bottom=report.get('source_footer_trimmed_at') or report['input_dimensions'][1]
    y1=max(box['y'],top);y2=min(box['y']+box['height'],bottom)
    if y2<=y1:return None
    return {**box,'y':y1-top,'height':y2-y1}


def project_region(item,source_box,report):
    box=project_box(source_box,report)
    if box is None:return None
    output=dict(item)
    if item.get('bbox'):
        projected=project_box(item['bbox'],report)
        if projected is None:return None
        output['bbox']=projected
    return box,output
