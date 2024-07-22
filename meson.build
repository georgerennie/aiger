project(
	'aiger++',
	'c',
	'cpp',
	license: 'MIT',
	default_options: [
		'c_std=gnu11',
		'cpp_std=c++20',
		'warning_level=3',
		'buildtype=release',
		'b_lto=true',
		'debug=true'
	]
)

install_headers('aiger++.hpp', 'aiger.h')

# It gets installed just as aiger.so... This means you can treat it just as
# regular aiger too. The distinction is that the c++ stuff is built in and a
# dependency on fmt
libaiger = library(
	'aiger',
	'aiger.c',
	'aiger++.cpp',
	include_directories: include_directories('.'),
	dependencies: dependency('fmt', required: true),
	install: true,
)

aiger = declare_dependency(
	link_with: libaiger,
	include_directories: include_directories('.'),
)
