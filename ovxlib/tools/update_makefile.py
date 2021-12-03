#!/usr/bin/python3.5
import os, re
import lxml.etree as ET
import uuid
import copy
from collections import OrderedDict

def find_files(folder, pattern):
    prog = re.compile(pattern)
    res = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if re.match(prog, name):
                item = os.path.join(root, name).replace('/','\\')
                res.append(item)
    return res

def build_data():
    header_folder = 'include'
    source_folder = 'src'
    header = sorted(find_files(header_folder, ".*\.h$"))
    source = sorted(find_files(source_folder, ".*\.c$"))
    return (header, source)

def update_vs_project(resources, vcxproj='ovxlib.vcxproj'):
    vcxproj_filters = vcxproj + '.filters'
    s_space = '  '
    def fill_tail(e, space=2, last=False):
        e.tail = '\n' + s_space*space if not last else '\n' + s_space*(space-1)
    def update_prj(resources, vxcproj):
        header,source = resources
        ns = {'ns': 'http://schemas.microsoft.com/developer/msbuild/2003'}
        vsproj = ET.parse(vxcproj, ET.XMLParser(ns_clean=True))
        root = vsproj.getroot()
        for item in root.findall('ns:ItemGroup', ns):
            child = item.find('ns:ClInclude', ns)
            if child is not None:
                item.clear(keep_tail=True)
                item.text = '\n' + s_space * 2
                for i,h in enumerate(header):
                    e = ET.Element('ClInclude', Include=h)
                    item.append(e)
                    fill_tail(e, 2, i == len(header) - 1)
                continue

            clcompile = item.findall('ns:ClCompile', ns)
            if len(clcompile) > 0:
                for child in clcompile:
                    if child.find('ns:ExcludedFromBuild', ns) is not None:
                        source.remove(child.get('Include'))
                        fill_tail(child, 2, False)
                    else:
                        item.remove(child)
                for s in source:
                    item.text = '\n' + s_space * 2
                    e = ET.Element('ClCompile', Include=s)
                    item.append(e)
                    fill_tail(e, 2, i == len(source) - 1)
                    e.text = '\n' + s_space * 3
                    e2 = ET.Element('ObjectFileName')
                    path = s.replace('src/','').replace('.c','.o')
                    path = path[:path.rfind('\\')+1]
                    e2.text = '$(IntDir)\\' + path
                    e.append(e2)
                    fill_tail(e2, 3, True)

        vsproj.write(vcxproj,
                pretty_print=True,encoding='utf-8',xml_declaration=True)

    def update_filters(resources, vcxproj_filters):
        header,source = resources
        ns = {'ns': 'http://schemas.microsoft.com/developer/msbuild/2003'}
        vsproj = ET.parse(vcxproj_filters, ET.XMLParser(ns_clean=True))
        root = vsproj.getroot()
        # build dir
        folders = list()
        for p in header + source:
            d = os.path.dirname(p.replace('\\', '/'))\
                    .replace('include', 'Header Files')\
                    .replace('src', 'Source Files')\
                    .replace('/','\\')
            prefix = ''
            for x in d.split("\\"):
                prefix += x
                folders.append(prefix)
                prefix += "\\"
        folders = sorted(set(folders))
        folders.remove('Header Files')
        folders.remove('Source Files')
        for item in root.findall('ns:ItemGroup', ns):
            # Update folders
            if item[0].tag == '{%s}%s'%(ns['ns'], 'Filter'):
                for child in item.findall('ns:Filter', ns):
                    p = child.get('Include')
                    if p in ['Source Files', 'Header Files', 'Resource Files']:
                        continue
                    #if p in folders:
                    #    folders.remove(p)
                    #    continue
                    item.remove(child)
                for i,folder in enumerate(folders):
                    e = ET.Element('Filter', Include=folder)
                    item.append(e)
                    e.text = '\n' + s_space * 3
                    fill_tail(e, 2, i == len(header) - 1)
                    e2 = ET.Element('UniqueIdentifier')
                    e2.text = '{%s}'%str(uuid.uuid3(uuid.NAMESPACE_DNS, folder))
                    e.append(e2)
                    fill_tail(e2, 3, True)
                item.text = '\n' + s_space * 2
                continue

            # Update includes
            child = item.find('ns:ClInclude', ns)
            if child is not None:
                item.clear(keep_tail=True)
                item.text = '\n' + s_space * 2
                for i,h in enumerate(header):
                    path = os.path.dirname(
                            h.replace('\\', '/').replace('include/','')).replace('/','\\')
                    path = 'Header Files\\' + path
                    if path[-1] == '\\':
                        path = path[:-1]
                    e = ET.Element('ClInclude', Include=h)
                    item.append(e)
                    e.text = '\n' + s_space * 3
                    fill_tail(e, 2, i == len(header) - 1)
                    e2 = ET.Element('Filter')
                    e2.text = path
                    e.append(e2)
                    fill_tail(e2, 3, True)
                continue

            # Update sources
            child = item.find('ns:ClCompile', ns)
            if child is not None:
                item.clear(keep_tail=True)
                item.text = '\n' + s_space * 2
                for i,s in enumerate(source):
                    path = os.path.dirname(
                            s.replace('\\', '/').replace('src/','')).replace('/','\\')
                    path = 'Source Files\\' + path
                    if path[-1] == '\\':
                        path = path[:-1]
                    e = ET.Element('ClCompile', Include=s)
                    item.append(e)
                    e.text = '\n' + s_space * 3
                    fill_tail(e, 2, i == len(header) - 1)
                    e2 = ET.Element('Filter')
                    e2.text = path
                    e.append(e2)
                    fill_tail(e2, 3, True)
                continue

        vsproj.write(vcxproj_filters,
                pretty_print=True,encoding='utf-8',xml_declaration=True)
    update_prj(copy.deepcopy(resources), vcxproj)
    update_filters(copy.deepcopy(resources), vcxproj_filters)




if __name__ == '__main__':
    resources = build_data()
    update_vs_project(resources, "ovxlib.vcxproj")
    update_vs_project(resources, "ovxlib.2012.vcxproj")

