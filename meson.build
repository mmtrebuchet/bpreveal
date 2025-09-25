project('bpreveal', 'c', 'fortran',
  default_options: ['buildtype=release'])

py = import('python').find_installation('python3')
f2py = find_program('f2py', required : true)
ext_suffix = py.get_variable('EXT_SUFFIX')

if ext_suffix == ''
  error('Could not determine Python extension suffix.')
endif

# Determine the package installation path
pkg_install_dir = py.get_install_dir(pure: false)

# Install the entire bpreveal directory structure with all Python files
install_subdir('src', install_dir: pkg_install_dir / 'bpreveal', strip_directory:true)

# Define internal directory path for compiled extensions
internal_install_dir = pkg_install_dir / 'bpreveal/internal'

# Custom targets to build extensions with f2py
custom_target('libjaccard',
  output: 'libjaccard' + ext_suffix,
  input: ['src/internal/libjaccard.c', 'src/internal/libjaccard.pyf'],
  command: [f2py, '-c', '@INPUT@', '-m', 'libjaccard', '--quiet'],
  install: true,
  install_dir: internal_install_dir,
  env: {'CFLAGS': '-fopenmp -fexpensive-optimizations -ffast-math -Ofast'}
)

custom_target('libslide',
  output: 'libslide' + ext_suffix,
  input: ['src/internal/libslide.c', 'src/internal/libslide.pyf'],
  command: [f2py, '-c', '-lgomp', '@INPUT@', '-m', 'libslide', '--quiet'],
  install: true,
  install_dir: internal_install_dir,
  env: {'CFLAGS': '-fopenmp -fexpensive-optimizations -ffast-math -Ofast'}
)

custom_target('libushuffle',
  output: 'libushuffle' + ext_suffix,
  input: ['src/internal/libushuffle.c', 'src/internal/libushuffle.pyf'],
  command: [f2py, '-c', '@INPUT@', '-m', 'libushuffle', '--quiet'],
  install: true,
  install_dir: internal_install_dir,
  env: {'CFLAGS': '-fopenmp -fexpensive-optimizations -ffast-math -Ofast'}
)

