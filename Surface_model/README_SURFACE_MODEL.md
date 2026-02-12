# Minimal Surface Forcing Model
This script implements a **minimal two-layer surface energy balance model**.

## 1. What this model does 

### 1.1 Physical model 
Two-layer surface:
- **Layer 1 (skin)**: fast-responding surface layer temperature `T1`
- **Layer 2 (subsurface)**: slow-responding subsurface temperature `T2`

Governing equations (per grid point):

$$
C_1 \frac{dT_1}{dt} = (1-\alpha)S - \sigma T_1^4 - k(T_1 - T_2)
$$

$$
C_2 \frac{dT_2}{dt} = k(T_1 - T_2)
$$

Where:

- S: insolation (W m⁻²)
- α: albedo 
- σ: Stefan–Boltzmann constant
- k: conductance between layers (W m⁻² K⁻¹)
- C₁, C₂: areal heat capacities (J m⁻² K⁻¹)

### 1.2 Insolation 
Insolation is computed from a simple solar-geometry model using **cos(SZA)** with an approximate solar declination from day-of-year (no equation-of-time correction).  


## 2. Requirements 
- Python 3
- torch

Example：
```bash
python -c "import torch; print(torch.__version__)"
```

## 3. Inputs 

### Required files 

`locations.csv`: defines region bounding box for each location_id

Expected columns:

Name, Latmin, Latmax, Lonmin, Lonmax

## 4. Usage 
```bash
# Basic run (default 18 hours)
python Surface_model.py ws-site1 --utc 2025-11-01T00:00

# Run exactly 1 day 
python Surface_model.py ws-site1 --utc 2025-11-01T00:00 --t_end 86400

# Control grid size 
python Surface_model.py ws-site1 --utc 2025-11-01T00:00 --nx 60 --ny 60

# Change output frequency 
# Default snapshots are every 6 hours (21600 s) to align with ERA 00/06/12/18.
python Surface_model.py ws-site1 --utc 2025-11-01T00:00 --save_every 3600
```
## 5. Command-line arguments 
```Markdown
Arg	                Meaning	                                        Default

--location_id	    region key in locations.csv                     required

--utc	            start time (ISO string, treated as UTC)         required

--nx	            number of grid points in longitude direction      60

--ny	            number of grid points in latitude direction       60

--dt	            time step in seconds                              60

--t_end	            integration length in seconds		            3 * 24 * 3600

--save_every	    snapshot interval in seconds		            21600

--Ts0	            initial temperature for both layers (K)	         250

--albedo	        surface albedo α	                             0.1

--C1	            skin areal heat capacity (J m⁻² K⁻¹)	         2e6

--C2	            subsurface areal heat capacity (J m⁻² K⁻¹)）	 2e7

--k	                layer coupling conductance (W m⁻² K⁻¹)	         15

--S0	            solar constant (W m⁻²)	                        1361

--locations_csv	    path to locations.csv		                    auto
```
## 6. Output
### 6.1 Output file

Results are saved to:

`./{location_id}/{location_id}_surface_buffers.pt`


Example:

`ws-site1/ws-site1_surface_buffers.pt`


This .pt file is a Python dictionary with keys:

`coords`

`buffers`

`params`


## 7. Output dictionary structure 
### 7.1 coords 
```
Key	            Meaning	                                Shape / Type
start_utc	    start time string	                    str
valid_time_utc	list of snapshot timestamps	            List[str] (Nt)
lead_seconds	seconds since start for each snapshot   Tensor (Nt)
lat_deg	        latitude grid		                    Tensor (ny,nx)
lon_deg	        longitude grid		                    Tensor (ny,nx)
```
### 7.2 buffers 

All buffers are stored as `torch.Tensor` on CPU.
```
Key	            Meaning		                            Unit	Shape
T1_K	        skin temperature (surface)		        K	    (Nt,ny,nx)
T2_K	        subsurface temperature		            K	    (Nt,ny,nx)
Tsfc_K	        alias of T1_K for BC		            K	    (Nt,ny,nx)
insol_Wm2	    incoming solar at TOA (toy)		        W/m²	(Nt,ny,nx)
dT1dt_Ks	    dT1/dt tendency		                    K/s	    (Nt,ny,nx)
dT2dt_Ks	    dT2/dt tendency	                        K/s	    (Nt,ny,nx)
Qsw_abs_Wm2	    absorbed shortwave (1-a)S		        W/m²	(Nt,ny,nx)
Qlw_up_Wm2	    upward thermal emission σT1^4		    W/m²	(Nt,ny,nx)
Qg_Wm2	        conductive flux k(T1-T2)		        W/m²	(Nt,ny,nx)
Qnet_Wm2	    net into layer1: Qsw - Qlw - Qg		    W/m²	(Nt,ny,nx)
Qatm_Wm2	    net to atmosphere (same as Qnet here)   W/m²	(Nt,ny,nx)
```
Sign convention for `Qg_Wm2 / Qg` 

`Qg > 0` means **downward** conduction (surface warmer than subsurface).


`Qg < 0` means **upward** conduction (subsurface warms the surface).


### 7.3 params 
```
Key	        Meaning 
dt	        timestep [s]	
t_end	    integration length [s]	
save_every	snapshot interval [s]	
S0	        solar constant [W/m²]	
albedo	    albedo α	
C1	        skin heat capacity [J m⁻² K⁻¹]	
C2	        subsurface heat capacity [J m⁻² K⁻¹]	
k	        coupling conductance [W m⁻² K⁻¹]	
bbox	    (latmin, latmax, lonmin, lonmax)	
```

## 8. Notes 

1. This model: no atmosphere, no clouds, no sensible/latent heat fluxes (H/LE).

2. Solar declination is an approximation; equation-of-time is ignored.

3. Results in the first day may still reflect spin-up from Ts0.

