#!/usr/bin/env python3

""" run script for static sequential microsynthesis """

import time
import microsimulation.static as Static
import microsimulation.utils as utils

# assert humanleague.version() > 1
DEFAULT_CACHE_DIR = "./cache"
DEFAULT_OUTPUT_DIR = "./data"


def main(params):
    """ Run it """

    resolution = params["resolution"]
    ref_year = params["census_ref_year"]
    horizon_year = params["horizon_year"]
    is_custom = params.get("custom_projection", False)
    variant = params["projection"]

    cache_dir = params["cache_dir"] if "cache_dir" in params else DEFAULT_CACHE_DIR
    output_dir = params["output_dir"] if "output_dir" in params else DEFAULT_OUTPUT_DIR

    use_fast_mode = params["mode"] == "fast"

    for region in params["regions"]:
        try:
            # start timing
            start_time = time.time()

            print("Static P Microsimulation: ", region, "@", resolution)

            # init microsynthesis
            ssm = Static.SequentialMicrosynthesis(region, resolution, variant, is_custom, cache_dir, output_dir, use_fast_mode)
            ssm.run(ref_year, horizon_year)

            print(region, "done. Exec time(s): ", time.time() - start_time)
        except RuntimeError as error:
            print(region, "FAILED: ", error)
    print("all done")


if __name__ == "__main__":

    PARAMS = utils.get_config()
    main(PARAMS)
