#Sort alphanumericcaly
import re

def alphanumeric_sort(name):
    parts = re.split('(\d+)', name)
    return [int(part) if part.isdigit() else part for part in parts]


#Extract sub_i
#Sort shapefile alphanumerically
