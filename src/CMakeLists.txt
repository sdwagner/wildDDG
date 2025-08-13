file(GLOB SOURCES ./Laplacians/*.cpp ./util/*.cpp ./util/marching_cubes/*.cpp)
file(GLOB HEADERS ./Laplacians/*.h ./util/*.h ./util/marching_cubes/*.h)

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