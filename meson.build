project(
	'aigerpp',
	'c',
	'cpp',
	license: 'MIT',
	default_options: [
		'c_std=gnu11',
		'cpp_std=c++20',
		'warning_level=0',
		'buildtype=release',
		'b_lto=true',
		'debug=true'
	]
)

install_headers('aigerpp.hpp', 'aiger.h')

fmt = dependency('fmt', required: true)

aigerpp_inc = include_directories('.')

# It gets installed just as aiger.so... This means you can treat it just as
# regular aiger too. The distinction is that the c++ stuff is built in and a
# dependency on fmt
aigerpp_lib = library(
	'aiger',
	'aiger.c',
	'aigerpp.cpp',
	include_directories: aigerpp_inc,
	dependencies: fmt,
	install: true,
)

aigerpp_dep = declare_dependency(
	link_with: aigerpp_lib,
	include_directories: aigerpp_inc,
)
