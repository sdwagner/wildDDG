file(GLOB SOURCES ./Laplacians/*.cpp ./util/*.cpp ./util/marching_cubes/*.cpp)
file(GLOB HEADERS ./Laplacians/*.h ./util/*.h ./util/marching_cubes/*.h)



if (EMSCRIPTEN)
    file(GLOB SOURCES_SMALL
        ./util/deformation_solver.cpp
        ./util/GeodesicsInHeat.cpp
        ./util/marching_cubes/*.cpp
        ./util/parameterization.cpp
        ./Laplacians/bunge_poly_laplace.cpp
        ./Laplacians/optimized_laplace.cpp)
    file(GLOB HEADERS_SMALL
        ./util/cylinder_generator.h
        ./util/deformation_solver.h
        ./util/hole_filling.h
        ./util/enums.h
        ./util/GeodesicsInHeat.h
        ./util/grid_generator.h
        ./util/mesh_converter.h
        ./util/parameterization.h
        ./util/marching_cubes/*.h
        ./Laplacians/bunge_poly_laplace.h
        ./Laplacians/optimized_laplace.h)
add_executable(web_demo web_demo.cpp ${HEADERS_SMALL} ${SOURCES_SMALL})
target_link_libraries(web_demo pmp_vis)
    set_target_properties(web_demo PROPERTIES LINK_FLAGS "--shell-file ${PROJECT_SOURCE_DIR}/data/shell.html --preload-file \"${CMAKE_SOURCE_DIR}/data/showcase_models@/\"")
else()
    add_executable(tri_tests run_tri_tests.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(tri_tests pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

    add_executable(unstructured_tri_tests run_unstructured_tri_tests.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(unstructured_tri_tests pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

    add_executable(poly_tests run_poly_tests.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(poly_tests pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

    add_executable(surface_viewer SurfaceViewer.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(surface_viewer pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

    add_executable(parameter_sweep parameter_sweep.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(parameter_sweep pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})

    add_executable(thingi10k_tests run_thingi10k_tests.cpp ${HEADERS} ${SOURCES})
    target_link_libraries(thingi10k_tests pmp_vis geometry-central OpenMeshCore OpenMeshTools geogram cinolib igl::core igl::predicates ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
endif ()
