"""Research motion interface: explicitly reject unlifted 2D as avatar XYZ."""
def require_avatar_motion(record):
    if record.get('coordinate_system')!='rig_3d_xyz':raise ValueError('A calibrated, validated 2D-to-3D stage is required before retargeting')
    if record.get('sign_language')!='fsl':raise ValueError('Expected France-LSF')
    if record.get('review_status')!='approved':raise ValueError('Motion review required')
    return record
