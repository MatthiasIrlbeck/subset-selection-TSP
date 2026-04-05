#include "run_cli.hpp"
#include "run_cli_internal.hpp"

int tsp_main_cli(int argc,char** argv){
    return run_cli_detail::tsp_main_cli_impl(argc, argv);
}
