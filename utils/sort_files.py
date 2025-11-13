import re

def alphanumeric_sort(name):
    """Sort filenames according a alphanumeric order"""
    
    parts = re.split('(\d+)', name)
    return [int(part) if part.isdigit() else part for part in parts]