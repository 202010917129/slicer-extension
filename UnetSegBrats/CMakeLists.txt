cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UnetSegBrats)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/UnetSegBrats")
set(EXTENSION_CATEGORY "AutoSeg")
set(EXTENSION_CONTRIBUTORS "MM (XUPT)")
set(EXTENSION_DESCRIPTION "This Extension use Unet model Brats20 data")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/UnetSegBrats.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/UnetSegBrats/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(UnetSegBrats)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
