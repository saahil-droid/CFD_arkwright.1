import numpy as np
import subprocess
import re 
from geometry import naca4_geometry, deflect_flap
import os


def write_dat_file(geo: dict, filename: str) -> None:

    # Reverse upper surface arrays to go from TE(trailing edge) to LE(leading edge), why ? because the lower surface is already from LE to TE, so we need to reverse the upper surface to make a continuous loop around the airfoil
    x_upper_reversed = geo['x_upper'][::-1]
    y_upper_reversed = geo['y_upper'][::-1]

    # Concatenate upper and lower surfaces, dropping the duplicate leading-edge point
    x_coords = np.concatenate([x_upper_reversed, geo['x_lower'][1:]])
    y_coords = np.concatenate([y_upper_reversed, geo['y_lower'][1:]])

    # Write to .dat file
    with open(filename, 'w') as dat_file:
        for x, y in zip(x_coords, y_coords):
            dat_file.write(f"{x:.10f} {y:.10f}\n")




def validate_xfoil_output(output: str, expected_cl: float, expected_cd: float) -> bool: 

    try:
        cl_value, cd_value = extract_cl_cd(output)

    except ValueError:
        return False

    return (
        np.isclose(cl_value, expected_cl, atol=1e-4)
        and
        np.isclose(cd_value, expected_cd, atol=1e-5)
    )


def extract_cl_cd(output: str) -> tuple[float, float]: 

    # Use regex to find lines containing CL and CD values
    cl_match = re.search(r"CL\s*=\s*([-+]?\d*\.\d+|\d+)", output)
    cd_match = re.search(r"CD\s*=\s*([-+]?\d*\.\d+|\d+)", output)

    if cl_match and cd_match:
        cl_value = float(cl_match.group(1)) #why 1, example, 1 would give 0.627 bu 0 would give CL = 0.627
        cd_value = float(cd_match.group(1))
        return cl_value, cd_value
    else:
        raise ValueError("Could not extract Cl and Cd from XFOIL output.")
    




def run_xfoil_with_timeout(dat_path: str, Re: float, alpha: float, timeout: int = 10) -> str:

    command = "\n".join([
        "PLOP",
        "G F",
        "",
        f"LOAD {dat_path}",
        "",
        "OPER",
        f"VISC {Re}",
        f"ALFA {alpha}",
        "QUIT",
    ])

    xfoil_path = os.path.expanduser(
        "~/xfoil/build/src/xfoil-6.97"
    )

    try:
        process = subprocess.Popen( #this is the command that runs xfoil in the background
            [xfoil_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate( #this communicates with the running xfoil process
            command,
            timeout=timeout
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"XFOIL failed with return code {process.returncode}:\n{stderr}"
            )

        return stdout

    except FileNotFoundError:
        raise RuntimeError(
            f"XFOIL executable not found at {xfoil_path}. "
            "Please check the path and ensure XFOIL is installed."
        )

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(
            "XFOIL process timed out."
        )




def sweep_xfoil(naca_code: str,n_panels: int,hinge_x: float,Re_values: list[float],alpha_values: list[float],flap_angles: list[float]) -> dict:

    results = {}

    for flap_angle in flap_angles:
        print(f"Generating geometry for flap angle {flap_angle} degrees")
        # Create base aerofoil
        geo = naca4_geometry(
            naca_code,
            n_panels
        )
        # Apply flap deflection
        geo_flapped = deflect_flap(
            geo,
            hinge_x=hinge_x,
            deflection_deg=flap_angle
        )
        # Save unique dat file
        dat_path = f"airfoil_flap_{flap_angle:+.1f}.dat"
        write_dat_file(
            geo_flapped,
            dat_path
        )

        for Re in Re_values:
            for alpha in alpha_values:

                try:
                    output = run_xfoil_with_timeout(
                        dat_path,
                        Re,
                        alpha
                    )

                    cl, cd = extract_cl_cd(output)

                    results[(Re, alpha, flap_angle)] = (cl, cd)

                except RuntimeError as e:
                    print(
                        f"Failed: "
                        f"Re={Re}, "
                        f"alpha={alpha}, "
                        f"flap={flap_angle}: {e}"
                    )

                    results[(Re, alpha, flap_angle)] = (
                        np.nan,
                        np.nan
                    )

    return results
