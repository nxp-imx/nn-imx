import os
import sys
import copy
from argparse import ArgumentParser

root = os.path.dirname(os.path.abspath(__file__))

_embedded_func = {
    'upper': lambda s: s.upper(),
    'lower': lambda s: s.lower()
    }

def load_file(path):
    path = os.path.join(root, path)
    with open(path, 'r') as f:
        return f.read()

def load_main_template(tmpl='kernel.tmpl'):
    path = os.path.join(root, 'template', tmpl)
    context = None
    with open(path, 'r') as f:
        context = f.read()
    return context

def gen_context(context, kernel_name, tmpl, name):
    gen_ctx = context \
           .replace('%KERNEL_META%', tmpl.kernel_meta) \
           .replace('%KERNEL_INITIALIZER%', tmpl.kernel_initializer) \
           .replace('%KERNEL_FUNCTION%', tmpl.kernel_function) \
           .replace('%KERNEL_QUERY%', tmpl.kernel_query) \
           .replace('%KERNEL_CHECK%', tmpl.kernel_check) \
           .replace('%KERNEL_TYPE%', name) \
           .replace('%upper(KERNEL_NAME)%', _embedded_func['upper'](kernel_name)) \
           .replace('%lower(KERNEL_TYPE)%', _embedded_func['lower'](name)) \
           .replace('%KERNEL_NAME%', kernel_name)
    return gen_ctx

def gen_source(context, kernel_name, tmpl, name, root_directory, vxc_directory):
    gen_ctx = gen_context(context, kernel_name, tmpl, name)
    fname = '%s_%s.c'%(kernel_name, name.lower())
    type_path = os.path.join(root_directory, name.lower())
    path = os.path.join(type_path, fname)
    with open(path, 'w') as f:
        f.write(gen_ctx)
    code_ctx = None
    if name == 'EVIS':
        fname = '%s.%s'%(kernel_name, 'vx')
        code_ctx = load_file('template/evis_code.tmpl')
    elif name == 'CL':
        fname = '%s.%s'%(kernel_name, name.lower())
        code_ctx = load_file('template/cl_code.tmpl')

    if code_ctx is not None:
        code_ctx = code_ctx.replace('%KERNEL_NAME%', kernel_name)
        path = os.path.join(vxc_directory, fname)
        with open(path, 'w') as f:
            f.write(code_ctx)

def main(kernel_name, is_custom):
    charset = list(range(ord('a'), ord('z') + 1, 1)) + list(range(ord('0'), ord('9') + 1, 1))
    charset = [chr(c) for c in charset] + [c for c in "_"]
    for c in kernel_name:
        if c not in charset:
            print("E kernel name contains invalid char '{}', only support [a-z0-9_]".format(c))
            return -1
    import template.kernel_cpu as tcpu
    import template.kernel_gpu as tgpu
    import template.kernel_evis as tevis
    license = load_file('template/license.tmpl')

    context = load_main_template()
    context = context.replace('%LICENSE%', license)

    if is_custom:
        root_directory = os.path.join(root, '..', 'src', 'custom', 'ops', 'kernel')
    else:
        root_directory = os.path.join(root, '..', 'src', 'kernel')
    evis_root_directory = os.path.join(root, '..', 'src', 'libnnext', 'ops', 'vx')
    cl_root_directory = os.path.join(root, '..', 'src', 'libnnext', 'ops', 'cl')
    gen_source(context, kernel_name, tevis, 'EVIS', root_directory, evis_root_directory)
    gen_source(context, kernel_name, tgpu, 'CL', root_directory, cl_root_directory)
    gen_source(context, kernel_name, tcpu, 'CPU', root_directory, cl_root_directory)

if __name__ == "__main__":
    options = ArgumentParser(description='Add a kernel into ovxlib')
    options.add_argument('kernel_name', help='Name for the kernel')
    options.add_argument('--type',
                         default='internal',
                         help='custom/internal')
    args = options.parse_args()
    main(args.kernel_name, args.type == 'custom')


