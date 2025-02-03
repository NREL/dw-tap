import re
import glob
from pathlib import Path
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd

import xarray as xr
import pickle as pkl
import h5pyd
import cfgrib

from shapely.geometry import Point

from tqdm.auto import tqdm

from rex.resource_extraction import MultiYearWindX

from dw_tap.data_fetching import getData, get_wtk_data_nn, get_wtk_data_idw, get_data_wtk_led_nn, get_data_era5_idw, get_data_bchrrr_idw

# The following allows finding data directory based on the config in ~/.tap.ini
import sys
sys.path.append("../scripts")
import dw_tap_data 
print("dw-tap data path set to:", dw_tap_data.path)


class SingletonABCMeta(ABCMeta, type):
    """A combined metaclass that handles both abstract base classes
    and singleton pattern.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            raise Exception(f"An instance of {cls.__name__} already exists.")
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
        return instance


class WindSiteType(metaclass=SingletonABCMeta):
    """A parent class that handles sets of wind sites (such as Met Tower, NPS,
    One Energy, etc). This class handles all functionality general to
    all wind site types.
    """
    def __init__(self, site_type: str, load=False):
        self.site_type = site_type
        self.data_path = Path(dw_tap_data.path) / self.site_type
        
        if load:
            try:
                self.load()
                return
            except Exception as e:
                print(e)
                print("Continuing with initialization.")
        
        self.initialize_metadata()
        
        self.sites = self.create_wind_sites()
        self.metadata = self.create_metadata_gdf()

        self.save()

    @abstractmethod
    def initialize_metadata(self):
        """
        Abstract method to initialize the site-type metadata.
        """
        pass
        

    @abstractmethod
    def create_wind_sites(self):
        """
        Abstract method to create new wind sites.
        """
        pass

    @abstractmethod
    def get_site_id(self, filename):
        """
        Abstract method to set the ID for each site.
        """
        pass

    @abstractmethod
    def get_site_data(self, filename):
        """
        Abstract method to get data for each site.
        """
        pass

    @abstractmethod
    def derive_site_metadata(self, filename, data):
        """
        Abstract method to derive metadata for each site.
        """
        pass

    def check_wtk_and_era5_overlap(self, time_start, time_end):
        """
        Checks if the time range of the readings from an individual wind site
        overlap with WTK and/or ERA5 model data time range.
        """
        wtk_start = pd.to_datetime("2007-01-01 00:00:00")
        wtk_end = pd.to_datetime("2013-12-31 11:59:59")
        wtk_overlap = min([time_end, wtk_end]) > max([time_start, wtk_start])
        
        era5_start = pd.to_datetime("2020-01-01 00:00:00")
        era5_end = pd.to_datetime("2023-09-01 00:00:00")
        era5_overlap = min([time_end, era5_end]) > max([time_start, era5_start])
        return wtk_overlap, era5_overlap

    def create_metadata_gdf(self):
        """
        Creates a Geopandas GeoDataFrame from the metadata
        derived for the wind sites when the sites were created.
        """
        df = pd.DataFrame([site.metadata for site in self.sites.values()])
        df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        gdf = gpd.GeoDataFrame(df,
                               geometry='geometry',
                               crs="EPSG:4326").sort_values(by='site_id').set_index('site_id')
        return gdf

    def get_wtk_data(self):
        # Open the wind data "file"
        # server endpoint, username, password are found in ~/.hscfg
        f = h5pyd.File("/nrel/wtk-us.h5", 'r', bucket="nrel-pds-hsds") 
        
        for site_id, site in tqdm(self.sites.items()):
            if not ('lat' in site.metadata and 'lon' in site.metadata):
                continue
            lat = site.metadata['lat']
            lon = site.metadata['lon']

            start_time = site.metadata['time_start'] if 'time_start' in site.metadata else None
            end_time = site.metadata['time_end'] if 'time_end' in site.metadata else None

            if 'height' in site.metadata: 
                # Just one height for this site
                print(f'Getting WTK data for site: {site_id}'.ljust(40), end='\r')
                height = site.metadata['height']
                
                site.wtk_data_interpolated = get_wtk_data_idw(f, lat, lon, height,
                                                     start_time=start_time, end_time=end_time, time_stride=1)
                # site.wtk_data_nn = get_wtk_data_nn(f, lat, lon, height,
                #                               start_time=start_time, end_time=end_time, time_stride=1)
            elif 'heights' in site.metadata: 
                # Multiple heights for this site
                site.wtk_data_interpolated = {}
                site.wtk_data_nn = {}
                for height in site.metadata['heights']:
                    print(f'Getting WTK data for site: {site_id} and height: {height}m'.ljust(80), end='\r')
                    site.wtk_data_interpolated[height] = get_wtk_data_idw(f, lat, lon, height,
                                                             start_time=start_time, end_time=end_time, time_stride=1)
                    # site.wtk_data_nn[height] = get_wtk_data_nn(f, lat, lon, height,
                    #                                       start_time=start_time, end_time=end_time, time_stride=1)
            else:
                continue

    
    def get_wtk_led_data(self, myr_pathstr):
        # Example myr_path: '/datasets/WIND/conus/v2.0.0/2020/conus_2020'
        # The height will be appended to the path in get_data_wtk_led_nn
        # Get dataset name
        myr_name = myr_pathstr.split('/')[-1]
        
        for site_id, site in tqdm(self.sites.items()):
            if not all(key in site.metadata for key in ['lat', 'lon', 'time_start', 'time_end']):
                continue
            lat = site.metadata['lat']
            lon = site.metadata['lon']
            start_time = site.metadata['time_start']
            end_time = site.metadata['time_end']
            if not hasattr(site, 'wtk_led_data_nn'):
                site.wtk_led_data_nn = {}

            if 'height' in site.metadata: 
                # Just one height for this site
                print(f'Getting WTK-LED data for site: {site_id}'.ljust(40), end='\r')
                height = site.metadata['height']
                site.wtk_led_data_interpolated[myr_name] = get_data_wtk_led_idw(myr_pathstr,
                                                                     lat, lon, height, 
                                                                     start_time=None, end_time=None, time_stride=None)
            elif 'heights' in site.metadata: 
                # Multiple heights for this site
                site.wtk_led_data_nn[myr_name] = {}
                for height in site.metadata['heights']:
                    print(f'Getting WTK-LED data for site: {site_id} and height: {height}m'.ljust(80), end='\r')
                    site.wtk_led_data_nn[myr_name][height] = get_data_wtk_led_nn(myr_pathstr,
                                                                       lat, lon, height, 
                                                                       start_time=None, end_time=None, time_stride=None)
            else:
                continue

    def get_era5_data(self):    
        dest_dir = Path(dw_tap_data.path.strip('~')) / "era5/conus"
        
        for site_id, site in self.sites.items():
            site.era5_data = {}
        
        for year in tqdm(range(2000,2024)):
            ds = xr.open_dataset(dest_dir / f"conus-{year}-hourly.grib", engine="cfgrib")
            
            for site_id, site in self.sites.items():
                if not all(col in site.metadata for col in ('lat','lon')):
                    continue
                if not any(col in site.metadata for col in ('height','heights')):
                    continue
                lat = site.metadata['lat']
                lon = site.metadata['lon']

                
                if 'heights' in site.metadata:
                    height = [height for height in site.metadata['heights']]
                    site.era5_data[year] = get_data_era5_idw(ds, lat, lon, height)
                elif 'height' in site.metadata:
                    height = site.metadata['height']
                    site.era5_data[year] = get_data_era5_idw(ds, lat, lon, height)

    def get_bchrrr_data(self, myr_pathstr):
        # Using the myr file directly here because this is just
        # one filename type.
        myr_name = myr_pathstr.split('/')[-1].split('.')[0]
        myr = MultiYearWindX(myr_pathstr, hsds=False)
        
        for site_id, site in tqdm(self.sites.items()):
            if not all(key in site.metadata for key in ['lat', 'lon', 'time_start', 'time_end']):
                continue
            lat = site.metadata['lat']
            lon = site.metadata['lon']
            start_time = site.metadata['time_start']
            end_time = site.metadata['time_end']
            if not hasattr(site, 'bchrrr_data'):
                site.bchrrr_data = {}

            if 'height' in site.metadata: 
                # Just one height for this site
                print(f'Getting BC-HRRR data for site: {site_id}'.ljust(40), end='\r')
                height = site.metadata['height']
                site.bchrrr_data[myr_name] = get_data_bchrrr_idw(myr, lat, lon, height,
                                                                 start_time=None, end_time=None, time_stride=None)
            elif 'heights' in site.metadata: 
                # Multiple heights for this site
                site.bchrrr_data[myr_name] = {}
                for height in site.metadata['heights']:
                    print(f'Getting BC-HRRR data for site: {site_id} and height: {height}m'.ljust(80), end='\r')
                    site.bchrrr_data[myr_name][height] = get_data_bchrrr_idw(myr, lat, lon, height, 
                                                                             start_time=None, end_time=None, time_stride=None)
            else:
                continue
            
        

    def load(self):
        load_path = self.data_path / f"{self.site_type}.pkl"
        try:
            with open(load_path, 'rb') as file:
                loaded_obj = pkl.load(file)
                self.__dict__.update(loaded_obj.__dict__)
            print(f"Loaded {self.site_type} object from {load_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {load_path} not found.")
        except Exception as e:
            raise Exception(f"Failed to load {load_path}: {e}")

    def save(self):
        save_path = self.data_path / f"{self.site_type}.pkl"
        with open(save_path, 'wb') as file:
            pkl.dump(self, file)
        print(f"{self.site_type} object saved to {save_path}")



class WindSite:
    """This class handles individual wind sites. For most wind sites,
    this will be a type of wind site (such as met tower or NPS), and a lat-lon
    location. Some wind sites (specifically met tower sites) have measurements at
    multiple heights.
    """
    def __init__(self, site_type: WindSiteType, filename: str):
        self.site_type = site_type
        self.filename = filename
        
        self.site_id = site_type.get_site_id(self.filename)
        self.data = site_type.get_site_data(self.filename)
        self.metadata = site_type.derive_site_metadata(self.site_id, self.data)  

        # Model data will be fetched for all sites simultaneously to avoid
        # excessive calls to the API.
        self.wtk_data_nn = None
        self.wtk_data_interpolated = None
        self.era5_data = None


class AdhocWindSites(WindSiteType):
    """This class handles all functionality specific to NPS wind sites.
    """
    # There is a top-level xlsx metadata file, and the site IDs
    # correspond with the names of the individual site csv files.
    def __init__(self, load=False):
        super().__init__("adhoc", load)

    def initialize_metadata(self):
        # Get initial wind site metadata from NPSTurbineData.xlsx
        metadata_fp = self.data_path / "metadata.csv"
        self._metadata = pd.read_csv(metadata_fp)
        self._metadata.set_index('site_id', inplace=True)
        self._metadata = self._metadata.to_dict('index')

    def create_wind_sites(self):
        sites = {}
        for site_id in self._metadata:
            site = WindSite(self, site_id)
            sites[site_id] = site
        return sites

    def get_site_id(self, site_id):
        return site_id
            
    def get_site_data(self, filename):
        return None

    def derive_site_metadata(self, site_id, data):
        site_metadata = self._metadata[site_id] if site_id in self._metadata else {}
        site_metadata['site_id'] = site_id
        return site_metadata

    def get_model_data(self):
        self.get_wtk_data()
        get_era5_data(self)
        


class NPSWindSites(WindSiteType):
    """This class handles all functionality specific to NPS wind sites.
    """
    # There is a top-level xlsx metadata file, and the site IDs
    # correspond with the names of the individual site csv files.
    def __init__(self, load=False):
        super().__init__("nps", load)

    def initialize_metadata(self):
        # Get initial wind site metadata from NPSTurbineData.xlsx
        metadata_fp = self.data_path / "NPSTurbineData.xlsx"
        self._metadata = pd.read_excel(metadata_fp).rename(columns={"Installed Product: Installed Product ID": 'site_id'})
        self._metadata['site_id'] = self._metadata['site_id'].astype(int)
        self._metadata.set_index('site_id', inplace=True)
        self._metadata = self._metadata[['Latitude', 
                                         'Longitude', 
                                         'Elevation (m)',
                                         'Country', 
                                         'Date Installed', 
                                         'Product Name', 
                                         'Rotor Diameter', 
                                         'Tower Height (Meters)', 
                                         'wind-diesel']].rename(columns={'Latitude': 'lat', 
                                                                         'Longitude': 'lon', 
                                                                         'Elevation (m)': 'elevation_meters', 
                                                                         'Country': 'country', 
                                                                         'Date Installed': 'insallation_date', 
                                                                         'Product Name': 'product_name', 
                                                                         'Rotor Diameter': 'rotor_diameter', 
                                                                         'Tower Height (Meters)': 'height'
                                                                        })
        self._metadata = self._metadata.to_dict('index')

    def create_wind_sites(self):
        sites = {}
        for f in tqdm(glob.glob("%s/*.csv" % self.data_path)):
            filename = Path(f).name     
            site = WindSite(self, filename)
            sites[site.site_id] = site
        return sites

    def get_site_id(self, filename):
        site_id = int(filename.split('.')[-2])
        return site_id
            
    def get_site_data(self, filename):
        data = pd.read_csv(self.data_path / filename)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
        return data

    def derive_site_metadata(self, site_id, data):
        site_metadata = self._metadata[site_id] if site_id in self._metadata else {}

        time_start = data.timestamp.min()
        time_end = data.timestamp.max()
        wtk_overlap, era5_overlap = self.check_wtk_and_era5_overlap(time_start, time_end)

        site_metadata.update({'wind_speed_min': data.wind_speed_mps.min(),
                              'wind_speed_med': data.wind_speed_mps.median(),
                              'wind_speed_max': data.wind_speed_max_mps.max(),
                              'wind_speed_mean': data.wind_speed_mps.mean(),
                              'wind_direction_mean': (data.yaw_position_deg % 360).mean(),
                              'air_temp_c_mean': data.ambient_temperature_degc.mean(),
                              'air_density_mean': data.air_density_kg_m3[data.air_density_kg_m3 > 0].mean(),
                              'time_start': time_start,
                              'time_end':  time_end,
                              'wtk_overlap': wtk_overlap,
                              'era5_overlap': era5_overlap,
                              'n_samples': len(data),
                              'site_id': site_id
                             })
        return site_metadata


class MetTowerWindSites(WindSiteType):
    """This class handles all functionality specific to Met Tower wind sites.
    """
    # There is no top-level xlsx file, metadata is derived from the individual
    # site filenames. Site data is found in individual csv files.
    def __init__(self, load=False):
        super().__init__("met_tower_data", load)

    def initialize_metadata(self):
        self._metadata = {}
        for f in glob.glob("%s/*.csv" % self.data_path):
            filename = Path(f).name

            if filename == 'tx_rincondelsanjose.ndbc.26.801.-97.471.ndbc.public.csv':
                # This file is misnamed, clean it by dropping the first instance of ndbc
                location, _, lat_deg, lat_min, lon_deg, lon_min, src, access = filename.split(".")[:8] # Dropping the .csv at the end
            else:
                location, lat_deg, lat_min, lon_deg, lon_min, src, access = filename.split(".")[:7] # Dropping the .csv at the end
            state, site = location.split('_')
            lat = float('.'.join([lat_deg, lat_min]))
            lon = float('.'.join([lon_deg, lon_min]))

            site_id = location
            self._metadata[site_id] = {'state': state,
                                       'site': site,
                                       'lat': lat,
                                       'lon': lon,
                                       'src': src,
                                       'access': access}

    def create_wind_sites(self):
        # Create a wind site for each site ID (get the data & derive site metatdata for each site)
        sites = {}
        for f in tqdm(glob.glob("%s/*.csv" % self.data_path)):
            filename = Path(f).name     
            site = WindSite(self, filename)
            sites[site.site_id] = site
        return sites

    def get_site_id(self, filename):
        site_id = filename.split('.')[0]
        return site_id
            
    def get_site_data(self, filename):
        data = pd.read_csv(self.data_path / filename)
        if 'Time' in data.columns:
            data.rename(columns={'Time': 'timestamp'}, inplace=True)
        elif 'Start Time' in data.columns:
            data.rename(columns={'Start Time': 'timestamp'}, inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
        return data

    def derive_site_metadata(self, site_id, data):
        site_metadata = self._metadata[site_id]
        
        time_start = data.timestamp.min()
        time_end = data.timestamp.max()
        wtk_overlap, era5_overlap = self.check_wtk_and_era5_overlap(time_start, time_end)

        # Get measurement heights by parsing the data column names
        pattern = r"(?:Spd|Dir|Temp)(\d+)m"
        heights = set()
        # There can be more than one column per speed
        speed_cols = defaultdict(list)
        dir_cols = defaultdict(list)
        temp_cols = defaultdict(list)
        for col in data.columns:
            matches = re.findall(pattern, col, re.IGNORECASE)
            height = {int(match) for match in matches}
            if len(height) == 0:
                continue
            heights.update(height)
            # Populate dictionaries for speed, direction, and temperature columns
            assert len(height) == 1 # There should only be one height per column
            height = list(height)[0] 
            if 'spd' in col.lower():
                speed_cols[height].append(col)
            elif 'dir' in col.lower():
                dir_cols[height].append(col)
            elif 'temp' in col.lower():
                temp_cols[height].append(col)

        # There are often multiple heights in a single dataset, so getting statistics 
        # for wind speed and direction could be confusing. Instead, we provide a list 
        # of the heights of the measurements in each dataset, and dictionaries for the 
        # columns with the wind speed/direction and air temperature for each height.
        site_metadata.update({'time_start':  time_start,
                              'time_end':  time_end,
                              'wtk_overlap': wtk_overlap,
                              'era5_overlap': era5_overlap,
                              'n_samples': len(data),
                              'fields': data.columns.to_list(),
                              'heights': heights,
                              'speed_cols': speed_cols,
                              'dir_cols': dir_cols,
                              'temp_cols': temp_cols,
                              'site_id': site_id.split('.')[0]
                             })
        return site_metadata


class OneEnergyWindSites(WindSiteType):
    """This class handles all functionality specific to One Energy wind sites.
    """
    # There is a top-level xlsx metadata file, and site data
    # is stored as multiple xlsx files in per-site directories
    def __init__(self, load=False):
        super().__init__("one_energy", load)

    def initialize_metadata(self):
        # Get initial wind site metadata from NPSTurbineData.xlsx
        metadata_fp = self.data_path / "OneEnergyTurbineData.xlsx"
        self._metadata = pd.read_excel(metadata_fp)
        # This line creates a site_id of the same format as the site directory name
        self._metadata['site_id'] = self._metadata.apply(lambda row: row['Public Site Name'].lower() + row['APRS ID'][2:], axis=1)
        self._metadata.set_index('site_id', inplace=True)
        self._metadata = self._metadata.rename(columns={'APRS ID': 'aprs_id',
                                                        'Public Site Name': 'site_name', 
                                                        'State': 'state',
                                                        'Model': 'model',
                                                        'Rotor Diameter (m)': 'rotor_diameter_meters',
                                                        'Latitude': 'lat',
                                                        'Longitude': 'lon',
                                                        'Hub Height (m)': 'height',
                                                        'Rating (kw)': 'rating_kw'
                                                       })
        self._metadata = self._metadata.to_dict('index')

    def create_wind_sites(self):
        sites = {}
        for site_dir in tqdm(self.data_path.iterdir(), total=len(list(self.data_path.iterdir()))):
            if site_dir.is_dir():
                # print(f"\nProcessing directory: {site_dir.name}")
                site = WindSite(self, site_dir)
                sites[site.site_id] = site
        return sites

    def get_site_id(self, site_dir):
        site_id = site_dir.name
        return site_id
            
    def get_site_data(self, site_dir):
        all_dfs = []
        for file_path in site_dir.glob("*.xlsx"):
            df = pd.read_excel(file_path)
            all_dfs.append(df)
        data = pd.concat(all_dfs, ignore_index=True)
        data.rename(columns={'Time': 'timestamp'}, inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
        return data

    def derive_site_metadata(self, site_id, data):
        site_metadata = self._metadata[site_id] if site_id in self._metadata else {}

        time_start = data.timestamp.min()
        time_end = data.timestamp.max()
        wtk_overlap, era5_overlap = self.check_wtk_and_era5_overlap(time_start, time_end)
        		
        site_metadata.update({'wind_speed_min': data.MinWindSpeed_m_s_.min(),
                              'wind_speed_med': data.AvgWindSpeed_m_s_.median(),
                              'wind_speed_max': data.MaxWindSpeed_m_s_.max(),
                              'wind_speed_mean': data.AvgWindSpeed_m_s_.mean(),
                              'time_start': time_start,
                              'time_end':  time_end,
                              'wtk_overlap': wtk_overlap,
                              'era5_overlap': era5_overlap,
                              'n_samples': len(data),
                              'site_id': site_id
                             })
        return site_metadata



class BergeyWindSites(WindSiteType):
    """This class handles all functionality specific to Bergey wind sites.
    """
    # There is a top-level xlsx metadata file, and site data
    # is stored as multiple xlsx files in per-site directories
    def __init__(self, load=False):
        super().__init__("bergey", load)

    def initialize_metadata(self):
        # Get initial wind site metadata from NPSTurbineData.xlsx
        metadata_fp = self.data_path / "bergeysitesNEW.csv"
        self._metadata = pd.read_csv(metadata_fp)
        # This line creates a site_id of the same format as the site directory name
        self._metadata['site_id'] = self._metadata['AID']
        self._metadata.set_index('site_id', inplace=True)
        self._metadata = self._metadata.rename(columns={'APRS ID': 'aprs_id',
                                                        'Public Site Name': 'public_site_name', 
                                                        'Internal Site Name	': 'internal_site_name',
                                                        'State': 'state',
                                                        'Site Type': 'site_type',
                                                        'Latitude': 'lat',
                                                        'Longitude': 'lon',
                                                        'Hub Height (m)': 'height',
                                                        'Site Notes': 'site_notes',
                                                        'Lidar Quality': 'lidar_quality',
                                                        'Lidar Collection Year': 'lidar_collection_year',
                                                        'Building Data Quality': 'building_data_quality',
                                                        'Turbine Periods with Consistent Generation Data': 'turbine_periods_w_consistent_generation_data',
                                                        'Measurement Privacy': 'privacy',
                                                        'Bergey Annual Average Wind Speed (m/s)': 'avg_annual_wind_speed_mps',
                                                        'Bergey Generation (kWh)': 'generation_kwh'
                                                       })
        self._metadata = self._metadata.to_dict('index')

    def create_wind_sites(self):
        sites = {}
        for site_id in self._metadata:
            site = WindSite(self, site_id)
            sites[site_id] = site
        return sites

    def get_site_id(self, site_dir):
        return site_dir
            
    def get_site_data(self, site_dir):
        return None

    def derive_site_metadata(self, site_id, data):
        self._metadata[site_id]['site_id'] = site_id
        return self._metadata[site_id]



