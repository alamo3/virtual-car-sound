cmake_minimum_required(VERSION 3.16)
project(headless_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(__forceinline=inline)
add_compile_options(-fpermissive -include ${CMAKE_CURRENT_SOURCE_DIR}/common.h)
set(PIRANHA_ENABLED ON)
include(FetchContent)
set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${CMAKE_CURRENT_SOURCE_DIR}/gtest)
add_subdirectory(dependencies/submodules/simple-2d-constraint-solver)
add_subdirectory(dependencies/submodules/csv-io)
add_subdirectory(dependencies/submodules/piranha)


add_library(engine-sim STATIC
    # Source files
    src/audio_buffer.cpp
    src/camshaft.cpp
    src/crankshaft.cpp
    src/combustion_chamber.cpp
    src/connecting_rod.cpp
    src/convolution_filter.cpp
    src/cylinder_bank.cpp
    src/cylinder_head.cpp
    src/delay_filter.cpp
    src/derivative_filter.cpp
    src/direct_throttle_linkage.cpp
    src/dynamometer.cpp
    src/engine.cpp
    src/exhaust_system.cpp
    src/feedback_comb_filter.cpp
    src/filter.cpp
    src/fuel.cpp
    src/function.cpp
    src/gas_system.cpp
    src/gaussian_filter.cpp
    src/governor.cpp
    src/ignition_module.cpp
    src/impulse_response.cpp
    src/intake.cpp
    src/jitter_filter.cpp
    src/leveling_filter.cpp
    src/low_pass_filter.cpp
    src/part.cpp
    src/piston.cpp
    src/piston_engine_simulator.cpp
    src/simulator.cpp
    src/standard_valvetrain.cpp
    src/starter_motor.cpp
    src/synthesizer.cpp
    src/throttle.cpp
    src/transmission.cpp
    src/utilities.cpp
    src/valvetrain.cpp
    src/vehicle.cpp
    src/vehicle_drag_constraint.cpp
    src/vtec_valvetrain.cpp

    # Include files
    include/audio_buffer.h
    include/application_settings.h
    include/camshaft.h
    include/crankshaft.h
    include/combustion_chamber.h
    include/connecting_rod.h
    include/convolution_filter.h
    include/cylinder_bank.h
    include/cylinder_head.h
    include/delay_filter.h
    include/derivative_filter.h
    include/direct_throttle_linkage.h
    include/dynamometer.h
    include/engine.h
    include/exhaust_system.h
    include/feedback_comb_filter.h
    include/filter.h
    include/fuel.h
    include/function.h
    include/gas_system.h
    include/gaussian_filter.h
    include/governor.h
    include/ignition_module.h
    include/impulse_response.h
    include/intake.h
    include/jitter_filter.h
    include/leveling_filter.h
    include/low_pass_filter.h
    include/part.h
    include/piston.h
    include/piston_engine_simulator.h
    include/simulator.h
    include/standard_valvetrain.h
    include/starter_motor.h
    include/synthesizer.h
    include/throttle.h
    include/transmission.h
    include/units.h
    include/utilities.h
    include/valvetrain.h
    include/vehicle.h
    include/vehicle_drag_constraint.h
    include/vtec_valvetrain.h
)

target_link_libraries(engine-sim
    simple-2d-constraint-solver
    csv-io
    )

target_include_directories(engine-sim
    PUBLIC dependencies/submodules)

if (PIRANHA_ENABLED)
    add_library(engine-sim-script-interpreter STATIC
        # Source files
        scripting/src/channel_types.cpp
        scripting/src/compiler.cpp
        scripting/src/engine_context.cpp
        scripting/src/language_rules.cpp

        # Include files
        scripting/include/actions.h
        scripting/include/camshaft_node.h
        scripting/include/channel_types.h
        scripting/include/compiler.h
        scripting/include/connecting_rod_node.h
        scripting/include/crankshaft_node.h
        scripting/include/cylinder_bank_node.h
        scripting/include/cylinder_head_node.h
        scripting/include/engine_context.h
        scripting/include/engine_node.h
        scripting/include/engine_sim.h
        scripting/include/exhaust_system_node.h
        scripting/include/intake_node.h
        scripting/include/function_node.h
        scripting/include/ignition_module_node.h
        scripting/include/ignition_wire_node.h
        scripting/include/impulse_response_node.h
        scripting/include/intake_node.h
        scripting/include/language_rules.h
        scripting/include/node.h
        scripting/include/object_reference_node.h
        scripting/include/object_reference_node_output.h
        scripting/include/piranha.h
        scripting/include/piston_node.h
        scripting/include/rod_journal_node.h
        scripting/include/standard_valvetrain_node.h
        scripting/include/transmission_node.h
        scripting/include/valvetrain_node.h
        scripting/include/vtec_valvetrain_node.h
		scripting/include/vehicle_node.h
    )

    target_include_directories(engine-sim-script-interpreter
        PUBLIC dependencies/submodules)

    target_link_libraries(engine-sim-script-interpreter
        engine-sim csv-io piranha)
endif (PIRANHA_ENABLED)


add_executable(headless headless.cpp)
target_link_libraries(headless PRIVATE engine-sim-script-interpreter engine-sim piranha csv-io simple-2d-constraint-solver)
