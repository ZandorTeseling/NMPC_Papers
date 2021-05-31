cmake_minimum_required(VERSION 3.0.0)
project(NMPC_PAPERS VERSION 0.1.0 LANGUAGES C CXX)

set(ACADOS_INSTALL_DIR ${PROJECT_SOURCE_DIR}/build/acados)
set(ACADOS_WITH_QPOASES ON)
set(ACADOS_PYTHON ON)
set(ACADOS_EXAMPLES ON)
set(BUILD_SHARED_LIBS ON)

add_subdirectory( ${PROJECT_SOURCE_DIR}/3rdParty )


file(GLOB DICSRETE_MODEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/second_order_ode_dt_model/second_order_ode_dt_dyn_disc_phi_fun.c
    ${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/second_order_ode_dt_model/second_order_ode_dt_dyn_disc_phi_fun_jac.c
    ${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/second_order_ode_dt_model/second_order_ode_dt_dyn_disc_phi_fun_jac_hess.c
    #${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/second_order_ode_dt_model/second_order_ode_dt_model.h
)

add_executable(discrete_sim  ${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/main_sim_second_order_ode_dt.c
                             ${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/acados_sim_solver_second_order_ode_dt.c
                             #${CMAKE_CURRENT_SOURCE_DIR}/MHE-DelayedSystem/c_generated_code/acados_sim_solver_second_order_ode_dt.h
                             ${DICSRETE_MODEL_SRC})
target_link_libraries(discrete_sim acados)

