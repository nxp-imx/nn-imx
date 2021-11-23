import os
import sys
import copy
from argparse import ArgumentParser

root = os.path.dirname(os.path.abspath(__file__))

_embedded_func = {
    'upper': lambda s: s.upper(),
    'lower': lambda s: s.lower()
    }

def get_blank(op, total_num):
    res = total_num - len(op)
    if res < 1:
        res = 1
    return res * ' '

def modify_ops_def(lines, op_name):
    lines.append('DEF_OP(' + op_name.upper() + ')\n')
    return lines

def modify_ovxlib_vcxproj(lines, op_name):
    new_lines = lines.copy()
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_tensorstackconcat.h') != -1:
            s = '    <ClInclude Include="include\\ops\\vsi_nn_op_%s.h" />\n'%op_name.lower()
            new_lines[index - 1] += s
            continue
        if line.find('vsi_nn_op_variable.c') != -1:
            s = '''\
    <ClCompile Include="src\\ops\\vsi_nn_op_%s.c" >\n\
      <ObjectFileName>$(IntDir)\\src\\ops\\</ObjectFileName>
    </ClCompile>
'''%(op_name.lower())
            new_lines[index - 1] += s
            continue
    return new_lines

def modify_ovxlib_vcxproj_filters(lines, op_name):
    new_lines = lines.copy()
    for index, line in enumerate(lines):
        if line.find('vsi_nn_vxkernel.h') != -1:
            s = '''\
    <ClInclude Include="include\\ops\\vsi_nn_op_%s.h">
      <Filter>Header Files\\ops</Filter>
    </ClInclude>
'''%op_name.lower()
            new_lines[index - 1] += s
            continue
        if line.find('vsi_nn_vxkernel.c') != -1:
            s = '''\
    <ClCompile Include="src\\ops\\vsi_nn_op_%s.c">
      <Filter>Source Files\\ops</Filter>
    </ClCompile>
'''%op_name.lower()
            new_lines[index - 1] += s
            continue
    return new_lines

def modify_vsi_nn_node_type_h(lines, op_name):
    new_lines = lines.copy()
    for index, line in enumerate(lines):
        if line.find('/* custom node head define define */') != -1:
            s = '#include "ops/vsi_nn_op_%s.h"\n'%op_name.lower()
            new_lines[index - 1] += s
        if line.find('client_param[128]') != -1:
            blank = get_blank(op_name, 19)
            s = '    vsi_nn_%s_param%s%s;\n'%(op_name.lower(), blank, op_name.lower())
            new_lines[index - 1] += s
    return new_lines

def modify_vsi_nn_code_generator_c(lines, op_name):
    new_lines = lines.copy()
    flag = False
    for index, line in enumerate(lines):
        if not flag and line.find('s_op_gen[]') != -1:
            flag = True
        elif flag and line.find('};') != -1:
            blank = get_blank(op_name, 22)
            s = '    /* %s */%sNULL,\n'%(op_name.upper(), blank)
            new_lines[index - 1] += s
            break
    return new_lines

modify_file_list = [
    {'file': '../include/interface/ops.def', 'func': modify_ops_def},
    {'file': '../ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '../ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '../ovxlib.2019.vcxproj', 'func': modify_ovxlib_vcxproj}
    {'file': '../ovxlib.2019.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '../include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    {'file': '../src/utils/vsi_nn_code_generator.c', 'func': modify_vsi_nn_code_generator_c},
]

modify_list_for_internal = [
    {'file': '../include/internal/internal_ops.def', 'func': modify_ops_def},
    {'file': '../ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '../ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '../ovxlib.2019.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '../ovxlib.2019.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '../include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    #{'file': '../src/utils/vsi_nn_code_generator.c', 'func': modify_vsi_nn_code_generator_c},
]

def modify_file(modify_list, op_name):
    for m in modify_list:
        file = os.path.join(root, m['file'])
        print("modify "+ file)
        with open(file, mode='r+', newline='\n', encoding='UTF-8') as fhndl:
            lines = m['func'](fhndl.readlines(),op_name)
            fhndl.seek(0)
            fhndl.writelines(lines)

def load_file(path):
    path = os.path.join(root, path)
    with open(path, 'r') as f:
        return f.read()

def load_main_template(op_name, tmpl='operation.tmpl'):
    path = os.path.join(root, 'template', tmpl)
    context = None
    with open(path, 'r') as f:
        context = f.read()
        context = context.replace('%upper(OP_NAME)%', _embedded_func['upper'](op_name))\
                       .replace('%lower(OP_NAME)%', _embedded_func['lower'](op_name))
    return context

def gen_source(path, context, license):
    gen_ctx = context.replace('%LICENSE%', license)
    with open(path, 'w') as f:
        f.write(gen_ctx)

def main(op_name):
    charset = list(range(ord('a'), ord('z') + 1, 1)) + list(range(ord('0'), ord('9') + 1, 1))
    charset = [chr(c) for c in charset] + [c for c in "_"]
    for c in op_name:
        if c not in charset:
            print("E operation name contains invalid char '{}', only support [a-z0-9_]".format(c))
            return -1
    license = load_file('template/license.tmpl')
    source = load_main_template(op_name)
    header = load_main_template(op_name, 'operation_header.tmpl')
    src_fname = 'vsi_nn_op_%s.c'%(op_name)
    hdr_fname = 'vsi_nn_op_%s.h'%(op_name)
    gen_source(os.path.join(root, '..', 'src', 'ops', src_fname), source, license)
    gen_source(os.path.join(root, '..', 'include', 'ops', hdr_fname), header, license)
    if (args.type == 'internal'):
        modify_file(modify_list_for_internal, op_name)
    else:
        modify_file(modify_file_list, op_name)

if __name__ == "__main__":
    options = ArgumentParser(description='Add an operation into ovxlib')
    options.add_argument('--type',
                         default='embedded',
                         help='embedded/internal')
    options.add_argument('op_name', help='Name for the operation')
    args = options.parse_args()
    main(args.op_name)


