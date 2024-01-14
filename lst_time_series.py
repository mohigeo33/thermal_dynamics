...
### About the code: This is a python script for building a time series of the land surface temperature (LST) from Landsat data.
### Most part of the code section in upto calculating of LST was used from the work of (Nill et. al., 2019). 
### However, the spectral indices, descriptive statistics extraction code is compiled by Mohiuddin, G.
### The statistical analysis code is also compiled by Mohiuddin	G.
...

import ee
ee.Initialize()


###################################################################### user input ######################################################################
# study period
year_start = 2000
year_end = 2021
month_start = 1
month_end = 12

# temperature threshold
t_threshold = 20

# cloud filter
max_cloud_cover = 60 # in percentage

# AOI
aoi =  '/content/drive/MyDrive/..../aoi.shp' # insert input here
roi = geemap.shp_to_ee(aoi)

# Min and max NDVI values
ndvi_v = 0.86
ndvi_s = -0.64

# emissivity value (Li et al., 2013)
epsilon_v = 0.985
epsilon_s = 0.97
epsilon_w = 0.99

# coefficients for atmospheric functions (Jiménez‐Muñoz et al. (2008) & Jiménez‐Muñoz et al. (2014))
cs_l8 = [0.04019, 0.02916, 1.01523,
         -0.38333, -1.50294, 0.20324,
         0.00918, 1.36072, -0.27514]
cs_l7 = [0.06518, 0.00683, 1.02717,
         -0.53003, -1.25866, 0.10490,
         -0.01965, 1.36947, -0.24310]
cs_l5 = [0.07518, -0.00492, 1.03189,
         -0.59600, -1.22554, 0.08104,
         -0.02767, 1.43740, -0.25844]
		 
		 
###################################################################### Functions ######################################################################
## Rename bands and clipping
# renaming bands
def fun_bands_l57(img):
       bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
       thermal_band = ['B6']
       new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
       new_thermal_bands = ['TIR']
       vnirswir = img.select(bands).multiply(0.0001).rename(new_bands)
       tir = img.select(thermal_band).multiply(0.1).rename(new_thermal_bands)
       return vnirswir.addBands(tir).copyProperties(img, ['system:time_start'])


def fun_bands_l8(img):
       bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
       thermal_band = ['B10']
       new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
       new_thermal_bands = ['TIR']
       vnirswir = img.select(bands).multiply(0.0001).rename(new_bands)
       tir = img.select(thermal_band).multiply(0.1).rename(new_thermal_bands)
       return vnirswir.addBands(tir).copyProperties(img, ['system:time_start'])

#clipping
def fun_clip(img):
  clip_img = img.clip(roi)
  return clip_img

## Masking
# Function to cloud mask Landsat TM, ETM+, OLI_TIRS Surface Reflectance Products (Foga et al., 2017)
def fun_mask_ls_sr(img):
       cloudShadowBitMask = ee.Number(2).pow(3).int()
       cloudsBitMask = ee.Number(2).pow(5).int()
       snowBitMask = ee.Number(2).pow(4).int()
       qa = img.select('pixel_qa')
       mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
              qa.bitwiseAnd(cloudsBitMask).eq(0)).And(
              qa.bitwiseAnd(snowBitMask).eq(0))
       return img.updateMask(mask)

# Function to mask LST below certain temperature threshold
def fun_mask_T(img):
    mask = img.select('LST').gt(t_threshold)
    return img.updateMask(mask)
	
## Matching and calibration
# Radiometric Calibration
def fun_radcal(img):
    radiance = ee.Algorithms.Landsat.calibratedRadiance(img).rename('RADIANCE')
    return img.addBands(radiance)

# L to ee.Image
def fun_l_addband(img):
    l = ee.Image(img.get('L')).select('RADIANCE').rename('L')
    return img.addBands(l)

# Create maxDifference-filter to match TOA and SR products
maxDiffFilter = ee.Filter.maxDifference(
    difference=2 * 24 * 60 * 60 * 1000,
    leftField= 'system:time_start',
    rightField= 'system:time_start'
)

# Define join: Water vapor
join_wv = ee.Join.saveBest(
    matchKey = 'WV',
    measureKey = 'timeDiff'
)

# Define join: Radiance
join_l = ee.Join.saveBest(
    matchKey = 'L',
    measureKey = 'timeDiff'
)

## Spectral Indices
# NDVI (Rouse et al., 1973)
def fun_ndvi(img):
    ndvi = img.normalizedDifference(['NIR', 'R']).rename('NDVI')
    return img.addBands(ndvi)

#MNDWI (Xu, 2006)
def fun_mndwi(img):
    mndwi = img.normalizedDifference(['G', 'SWIR1']).rename('MNDWI')
    return img.addBands(mndwi)

# IBI (Xu, 2008)
def fun_ibi(img):
    const = ee.Number(2)

    ibi = img.expression(
        '(((const * swir1) / (swir1 + nir))-((nir / (nir + red)) + (green / (green + swir1))))   / (((const * swir1) / (swir1 + nir))+((nir / (nir + red)) + (green / (green + swir1))))',
        {
            'swir1': img.select('SWIR1'),
            'nir': img.select('NIR'),
            'red': img.select('R'),
            'green': img.select('G'),
            'const': const
        }).rename('IBI')
    return img.addBands(ibi)
	
## Parameter calculation
# Fraction Vegetation Cover (FVC) (Carlson & Ripley, 1997)
def fun_fvc(img):
    fvc = img.expression(
        '((NDVI-NDVI_s)/(NDVI_v-NDVI_s))**2',
        {
            'NDVI': img.select('NDVI'),
            'NDVI_s': ndvi_s,
            'NDVI_v': ndvi_v
        }
    ).rename('FVC')
    return img.addBands(fvc)

# emissivity (Sobrino et al., 2008)
# scale Emissivity (Epsilon) between NDVI_s and NDVI_v
def fun_epsilon_scale(img):
    epsilon_scale = img.expression(
        'epsilon_s+(epsilon_v-epsilon_s)*FVC',
        {
            'FVC': img.select('FVC'),
            'epsilon_s': epsilon_s,
            'epsilon_v': epsilon_v
        }
    ).rename('EPSILON_SCALE')
    return img.addBands(epsilon_scale)


# emissivity (Epsilon)
def fun_epsilon(img):
    pseudo = img.select(['NDVI']).set('system:time_start', img.get('system:time_start'))
    epsilon = pseudo.where(img.expression('NDVI > NDVI_v',
                                          {'NDVI': img.select('NDVI'),
                                           'NDVI_v': ndvi_v}), epsilon_v)
    epsilon = epsilon.where(img.expression('NDVI < NDVI_s && NDVI >= 0',
                                           {'NDVI': img.select('NDVI'),
                                            'NDVI_s': ndvi_s}), epsilon_s)
    epsilon = epsilon.where(img.expression('NDVI < 0',
                                           {'NDVI': img.select('NDVI')}), epsilon_w)
    epsilon = epsilon.where(img.expression('NDVI <= NDVI_v && NDVI >= NDVI_s',
                                           {'NDVI': img.select('NDVI'),
                                            'NDVI_v': ndvi_v,
                                            'NDVI_s': ndvi_s}), img.select('EPSILON_SCALE')).rename('EPSILON')
    return img.addBands(epsilon)

# Function to scale WV content product (Lantz et al., 2010)
def fun_wv_scale(img):
    wv_scaled = ee.Image(img.get('WV')).multiply(0.1).rename('WV_SCALED')
    wv_scaled = wv_scaled.resample('bilinear')
    return img.addBands(wv_scaled)
	
## Land surface temperature calculation
# Atmospheric Functions (Lantz et al., 2010 & Jiménez-Muñoz et al., 2014)
def fun_af1(cs):
    def wrap(img):
        af1 = img.expression(
            '('+str(cs[0])+'*(WV**2))+('+str(cs[1])+'*WV)+('+str(cs[2])+')',
            {
                'WV': img.select('WV_SCALED')
            }
        ).rename('AF1')
        return img.addBands(af1)
    return wrap


def fun_af2(cs):
    def wrap(img):
        af2 = img.expression(
            '('+str(cs[3])+'*(WV**2))+('+str(cs[4])+'*WV)+('+str(cs[5])+')',
            {
                'WV': img.select('WV_SCALED')
            }
        ).rename('AF2')
        return img.addBands(af2)
    return wrap

def fun_af3(cs):
    def wrap(img):
        af3 = img.expression(
            '('+str(cs[6])+'*(WV**2))+('+str(cs[7])+'*WV)+('+str(cs[8])+')',
            {
                'WV': img.select('WV_SCALED')
            }
        ).rename('AF3')
        return img.addBands(af3)
    return wrap

# Gamma Functions
def fun_gamma_l8(img):
    gamma = img.expression('(BT**2)/(1324*L)',
                           {'BT': img.select('TIR'),
                            'L': img.select('L')
                            }).rename('GAMMA')
    return img.addBands(gamma)


def fun_gamma_l7(img):
    gamma = img.expression('(BT**2)/(1277*L)',
                           {'BT': img.select('TIR'),
                            'L': img.select('L')
                            }).rename('GAMMA')
    return img.addBands(gamma)


def fun_gamma_l5(img):
    gamma = img.expression('(BT**2)/(1256*L)',
                           {'BT': img.select('TIR'),
                            'L': img.select('L')
                            }).rename('GAMMA')
    return img.addBands(gamma)


# Delta Functions
def fun_delta_l8(img):
    delta = img.expression('BT-((BT**2)/1324)',
                           {'BT': img.select('TIR')
                            }).rename('DELTA')
    return img.addBands(delta)


def fun_delta_l7(img):
    delta = img.expression('BT-((BT**2)/1277)',
                           {'BT': img.select('TIR')
                            }).rename('DELTA')
    return img.addBands(delta)


def fun_delta_l5(img):
    delta = img.expression('BT-((BT**2)/1256)',
                           {'BT': img.select('TIR')
                            }).rename('DELTA')
    return img.addBands(delta)


# Land Surface Temperature (Jimenez-Munoz et al., 2008)
def fun_lst(img):
    lst = img.expression(
        '(GAMMA*(((1/EPSILON)*(AF1*L+AF2))+AF3)+DELTA)-273.15',
        {
            'GAMMA': img.select('GAMMA'),
            'DELTA': img.select('DELTA'),
            'EPSILON': img.select('EPSILON'),
            'AF1': img.select('AF1'),
            'AF2': img.select('AF2'),
            'AF3': img.select('AF3'),
            'L': img.select('L')
        }
    ).rename('LST')
    return img.addBands(lst)

def fun_mask_lst(img):
    mask = img.select('LST').gt(t_threshold)
    return img.updateMask(mask)
	

## Descriptive statistics extraction and date variables
#function for extracting image info
def fctn_get_image_stats(img):
  img_lst = img.select('LST')

  img_lst_mean_value = img_lst.reduceRegion(ee.Reducer.mean(), roi, 30,  crs = 'EPSG:32648', bestEffort = True, maxPixels = 1e9).getInfo()['LST']
  img_lst_max_value = img_lst.reduceRegion(ee.Reducer.max(), roi, 30,  crs = 'EPSG:32648', bestEffort = True, maxPixels = 1e9).getInfo()['LST']
  img_lst_min_value = img_lst.reduceRegion(ee.Reducer.min(), roi, 30,  crs = 'EPSG:32648', bestEffort = True, maxPixels = 1e9).getInfo()['LST']

  img_date = img.date().getInfo()['value']

  img_sytemindex = img.get('system:index').getInfo()

  img_cloud_cover = img.get('L')
  img_cloud_cover = ee.Image(img_cloud_cover)
  img_cloud_cover = img_cloud_cover.get('CLOUD_COVER').getInfo()
  # print(img_cloud_cover)

  img_all_info = {
      'system:index': img_sytemindex,
      'date': img_date,
      'mean_lst' : img_lst_mean_value,
      'min_lst' : img_lst_min_value,
      'max_lst' : img_lst_max_value,
      'cloud_cover': img_cloud_cover
  }
  return img_all_info

# Function to add date variables to DataFrame.
def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['Millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df
  
###################################################################### Calling the image collections ######################################################################
# Landsat 5 TM
imgCol_L5_TOA = ee.ImageCollection('LANDSAT/LT05/C01/T1')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .select(['B6'])

imgCol_L5_SR = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .map(fun_mask_ls_sr)

imgCol_L5_SR = imgCol_L5_SR.map(fun_bands_l57)

# Landsat 7 ETM+
imgCol_L7_TOA = ee.ImageCollection('LANDSAT/LE07/C01/T1')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .select(['B6_VCID_2'])

imgCol_L7_SR = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .map(fun_mask_ls_sr)

imgCol_L7_SR = imgCol_L7_SR.map(fun_bands_l57)

# Landsat 8 OLI-TIRS
imgCol_L8_TOA = ee.ImageCollection('LANDSAT/LC08/C01/T1')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .select(['B10'])

imgCol_L8_SR = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))\
    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))\
    .map(fun_mask_ls_sr)

imgCol_L8_SR = imgCol_L8_SR.map(fun_bands_l8)

# NCEP/NCAR Water Vapor Product
imgCol_WV = ee.ImageCollection('NCEP_RE/surface_wv')\
    .filterBounds(roi)\
    .filter(ee.Filter.calendarRange(year_start,year_end,'year'))\
    .filter(ee.Filter.calendarRange(month_start,month_end,'month'))


###################################################################### Calculation of LST and other indices ######################################################################
# TOA (Radiance) and SR
imgCol_L5_TOA = imgCol_L5_TOA.map(fun_radcal)
imgCol_L7_TOA = imgCol_L7_TOA.map(fun_radcal)
imgCol_L8_TOA = imgCol_L8_TOA.map(fun_radcal)

imgCol_L5_SR = ee.ImageCollection(join_l.apply(imgCol_L5_SR, imgCol_L5_TOA, maxDiffFilter))
imgCol_L7_SR = ee.ImageCollection(join_l.apply(imgCol_L7_SR, imgCol_L7_TOA, maxDiffFilter))
imgCol_L8_SR = ee.ImageCollection(join_l.apply(imgCol_L8_SR, imgCol_L8_TOA, maxDiffFilter))

imgCol_L5_SR = imgCol_L5_SR.map(fun_l_addband)
imgCol_L7_SR = imgCol_L7_SR.map(fun_l_addband)
imgCol_L8_SR = imgCol_L8_SR.map(fun_l_addband)

# Water Vapor
imgCol_L5_SR = ee.ImageCollection(join_wv.apply(imgCol_L5_SR, imgCol_WV, maxDiffFilter))
imgCol_L7_SR = ee.ImageCollection(join_wv.apply(imgCol_L7_SR, imgCol_WV, maxDiffFilter))
imgCol_L8_SR = ee.ImageCollection(join_wv.apply(imgCol_L8_SR, imgCol_WV, maxDiffFilter))

imgCol_L5_SR = imgCol_L5_SR.map(fun_wv_scale)
imgCol_L7_SR = imgCol_L7_SR.map(fun_wv_scale)
imgCol_L8_SR = imgCol_L8_SR.map(fun_wv_scale)

# Atmospheric Functions
imgCol_L5_SR = imgCol_L5_SR.map(fun_af1(cs_l5))
imgCol_L5_SR = imgCol_L5_SR.map(fun_af2(cs_l5))
imgCol_L5_SR = imgCol_L5_SR.map(fun_af3(cs_l5))

imgCol_L7_SR = imgCol_L7_SR.map(fun_af1(cs_l7))
imgCol_L7_SR = imgCol_L7_SR.map(fun_af2(cs_l7))
imgCol_L7_SR = imgCol_L7_SR.map(fun_af3(cs_l7))

imgCol_L8_SR = imgCol_L8_SR.map(fun_af1(cs_l8))
imgCol_L8_SR = imgCol_L8_SR.map(fun_af2(cs_l8))
imgCol_L8_SR = imgCol_L8_SR.map(fun_af3(cs_l8))

# Delta and Gamma Functions
imgCol_L5_SR = imgCol_L5_SR.map(fun_delta_l5)
imgCol_L7_SR = imgCol_L7_SR.map(fun_delta_l7)
imgCol_L8_SR = imgCol_L8_SR.map(fun_delta_l8)

imgCol_L5_SR = imgCol_L5_SR.map(fun_gamma_l5)
imgCol_L7_SR = imgCol_L7_SR.map(fun_gamma_l7)
imgCol_L8_SR = imgCol_L8_SR.map(fun_gamma_l8)

# Merge Collections
imgCol_merge = imgCol_L8_SR.merge(imgCol_L7_SR).merge(imgCol_L5_SR)
imgCol_merge = imgCol_merge.sort('system:time_start')

# Clipping to ROI
imgCol_merge = imgCol_merge.map(fun_clip)

# Parameters and Indices
imgCol_merge = imgCol_merge.map(fun_ndvi)
imgCol_merge = imgCol_merge.map(fun_ibi)
imgCol_merge = imgCol_merge.map(fun_mndwi)
imgCol_merge = imgCol_merge.map(fun_fvc)
imgCol_merge = imgCol_merge.map(fun_epsilon_scale)
imgCol_merge = imgCol_merge.map(fun_epsilon)

# LST
imgCol_merge = imgCol_merge.map(fun_lst)
imgCol_merge = imgCol_merge.map(fun_mask_lst)

###################################################################### Descriptive statistics extraction ######################################################################
# converting image collection to list
doi = imgCol_merge
doiList = doi.toList(doi.size())
doiList_size = doiList.size().getInfo()
print('Total Images in Data of Interest (doi) dataset: ',doiList_size)

# creating the dataframe
df = pd.DataFrame(columns=['SystemIndex', 'Millis', 'MeanLST', 'MaxLST', 'MinLST', 'Cloud'])
# iteration
for i in tqdm(range(doiList_size)):
  image = ee.Image(doiList.get(i))
  image_info = fctn_get_image_stats(image)

  df = df.append({'SystemIndex': image_info['system:index'],
                  'Millis':  image_info['date'],
                  'MeanLST': image_info['mean_lst'],
                  'MaxLST': image_info['max_lst'],
                  'MinLST': image_info['min_lst'],
                  'Cloud' : image_info['cloud_cover']
                  }, ignore_index=True)
                  
#adding the date variables
df = add_date_info(df)

# Converting the variables into numeric
df["MeanLST"] = pd.to_numeric(df["MeanLST"])
meanlst = df["MeanLST"]
df["MaxLST"] = pd.to_numeric(df["MaxLST"])
maxlst = df["MaxLST"]
df["MinLST"] = pd.to_numeric(df["MinLST"])
minlst = df["MinLST"]
df['Year'] = pd.to_numeric(df['Year'])
df['Month'] = pd.to_numeric(df['Month'])
df['Cloud'] = pd.to_numeric(df['Cloud'])

# Adding sensor information
df = df.assign(Category=df['SystemIndex'].str.slice(stop=8))
df['Category'] = df['Category'].str.replace('2_LT05_1', 'LS5')
df['Category'] = df['Category'].str.replace('1_2_LE07', 'LS7')
df['Category'] = df['Category'].str.replace('1_1_LC08', 'LS8')
df['Sensor'] = df['Category']
df = df.drop(columns=['Category'])

# drop the miscalculated lst data
df = df[df['SystemIndex'] != '2_LT05_126052_20010921']

# export the dataframe to a CSV file (optional)
df.to_csv('lst.csv', index=False)

###################################################################### Statistical analysis ######################################################################
# Sensor wise distribution of selected images
chart = alt.Chart(df).mark_point(size=50, filled=True).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('DOY:Q', title='Day of year'),
    color=alt.Color(
        'Sensor:N',
        legend=alt.Legend(title="Landsat")
    ),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('DOY:Q', title='Day of year'),
        alt.Tooltip('sensor:N', title='Satellite type')
    ]
).properties(width=600, height=300)

chart

# boxplot of minimum, mean and maximum LST
fig, ax = plt.subplots()
boxplot = df.boxplot(column=['MinLST', 'MeanLST', 'MaxLST'], ax=ax, patch_artist=True, return_type='dict')
ax.set_ylabel('LST in °C')
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', label='MinLST'),
                   Patch(facecolor='lightgreen', label='MeanLST'),
                   Patch(facecolor='lightcoral', label='MaxLST')]
ax.legend(handles=legend_elements, loc='upper left')

plt.show()

# mean LST calendar heatmap
alt.Chart(df).mark_rect().encode(
    x='Year:O',
    y='Month:O',
    color=alt.Color(
        'MeanLST:Q', title = 'Mean LST (in °C)', scale=alt.Scale(scheme='viridis', domain=(20,45 ))),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Month:O', title='Month'),
        alt.Tooltip('MeanLST:Q', title='LST')
    ]).properties(width=600, height=300)


# Create boxplot with annual LST range
df_melted = df.melt(id_vars='Year', value_vars=['MinLST', 'MeanLST', 'MaxLST'],
                      var_name='LST Type', value_name='Temperature')
chart = alt.Chart(df_melted).mark_boxplot().encode(
    x='Year:O',
    y=alt.Y('Temperature:Q', scale=alt.Scale(domain=(20, df2_melted['Temperature'].max())), title='LST in °C'),
    color='LST Type:N',
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('Temperature:Q', title='Temperature'),
        alt.Tooltip('LST Type:N', title='LST Type')
    ]
).properties(
    width=600,
    height=300,
    title=''
)

chart

# LST trend
df_trend = df.interpolate()
base = alt.Chart(df_trend).encode(
    x=alt.X('Timestamp:T', title='Year'),
    tooltip=[
        alt.Tooltip('Timestamp:T', title='Year')
    ]
)

mean_line = base.mark_line(color='red').encode(
    y=alt.Y('MeanLST:Q', scale=alt.Scale(domain=[20, 70]), title='LST in °C'),
    tooltip=alt.Tooltip('MeanLST:Q'),
    color=alt.value('red')
)

min_line = base.mark_line(color='violet').encode(
    y=alt.Y('MinLST:Q', scale=alt.Scale(domain=[20, 70])),
    tooltip=alt.Tooltip('MinLST:Q'),
    color=alt.value('violet')
)

max_line = base.mark_line(color='grey').encode(
    y=alt.Y('MaxLST:Q', scale=alt.Scale(domain=[20, 70])),
    tooltip=alt.Tooltip('MaxLST:Q'),
    color=alt.value('grey')
)

chart = (mean_line + min_line + max_line).properties(width=600, height=300).interactive()

legend = alt.Chart(pd.DataFrame({
    'LST Type': ['MeanLST', 'MinLST', 'MaxLST'],
    'Color': ['red', 'violet', 'grey']
})).mark_circle(size=100).encode(
    y=alt.Y('LST Type:N', axis=alt.Axis(orient='right')),
    color=alt.Color('Color:N', scale=None, legend=None)
)

(chart | legend)

###################################################################### Correlation analysis ######################################################################
# function for loading the csv/txt file
def load_csv(filepath):
    data =  []
    col = []
    checkcol = False
    with open(filepath) as f:
        for val in f.readlines():
            val = val.replace("\n","")
            val = val.split(',')
            if checkcol is False:
                col = val
                checkcol = True
            else:
                data.append(val)
    df = pd.DataFrame(data=data, columns=col)
    return df
Data2 = load_csv('cor_points.csv')
df3 = Data2

df3["NDVI"] = pd.to_numeric(df3["NDVI"])
df3["IBI"] = pd.to_numeric(df3["IBI"])
df3["MNDWI"] = pd.to_numeric(df3["MNDWI"])
df3["LST"] = pd.to_numeric(df3["LST"])

# correlation matrix
dfndvi = df3[(df3["NDVI"] >= 0.2)]
dfndvi
dfcor2 = dfndvi[['LST', 'NDVI']]
cor_data2 = (dfcor2
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data2['correlation_label'] = cor_data2['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
cor_data2


dfmndwi = df3[(df3["NDVI"] < 0.2)]
dfmndwi
dfcor3 = dfmndwi[['LST', 'MNDWI']]
cor_data3 = (dfcor3
              .corr().stack()
              .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data3['correlation_label'] = cor_data3['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
cor_data3

# Calculate individual correlation values
cor_LST_NDVI = dfndvi[['LST', 'NDVI']].corr().loc['LST', 'NDVI']
cor_LST_MNDWI = dfmndwi[['LST', 'MNDWI']].corr().loc['LST', 'MNDWI']
cor_LST_IBI = df3[['LST', 'IBI']].corr().loc['LST', 'IBI']
cor_NDVI_IBI = df3[['NDVI', 'IBI']].corr().loc['NDVI', 'IBI']
cor_NDVI_MNDWI = df3[['NDVI', 'MNDWI']].corr().loc['NDVI', 'MNDWI']
cor_IBI_MNDWI = df3[['IBI', 'MNDWI']].corr().loc['IBI', 'MNDWI']

# Create a new dataframe with the desired correlations
cor_matrix = pd.DataFrame({'LST': [1, cor_LST_NDVI, cor_LST_IBI, cor_LST_MNDWI],
                           'NDVI': [cor_LST_NDVI, 1, cor_NDVI_IBI, cor_NDVI_MNDWI],
                           'IBI': [cor_LST_IBI, cor_NDVI_IBI, 1, cor_IBI_MNDWI],
                           'MNDWI': [cor_LST_MNDWI, cor_NDVI_MNDWI, cor_IBI_MNDWI, 1]},
                          index=['LST', 'NDVI', 'IBI', 'MNDWI'])

# Display the combined correlation matrix
cor_matrix

# Convert the correlation matrix to a long format
cor_data_combined = cor_matrix.stack().reset_index().rename(columns={'level_0': 'variable', 'level_1': 'variable2', 0: 'correlation'})

# Add the correlation_label column
cor_data_combined['correlation_label'] = cor_data_combined['correlation'].map('{:.2f}'.format)

# Altair chart
base = alt.Chart(cor_data_combined).encode(
    x='variable2:O',
    y='variable:O'
)

# Text layer with correlation labels
# Colors are for easier readability
text = base.mark_text().encode(
    text='correlation_label',
    color=alt.condition(
        alt.datum.correlation > 0.5,
        alt.value('white'),
        alt.value('black')
    )
)

# The correlation heatmap itself
cor_plot = base.mark_rect().encode(
    color='correlation:Q'
).properties(width=600, height=300)

cor_plot + text  # The '+' means overlaying the text and rect layer

## Scatter plots between LST and spectral indices
# LST vs IBI
clean_data = df3.dropna(subset=['IBI', 'LST'])

x = clean_data['IBI']
y = clean_data['LST']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
     fontsize=11, verticalalignment='top')

plt.title('IBI vs LST')
plt.xlabel('IBI')
plt.ylabel('LST')
plt.figure(figsize=(5, 4))
plt.show()

# LST vs MNDWI
clean_data2 = df3.dropna(subset=['MNDWI', 'LST'])

x = clean_data2['MNDWI']
y = clean_data2['LST']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.60, 0.95, text, transform=plt.gca().transAxes,
     fontsize=11, verticalalignment='top')

plt.title('MNDWI vs LST')
plt.xlabel('MNDWI')
plt.ylabel('LST')
plt.figure(figsize=(5, 4))
plt.show()

# LST vs MNDWI (without the vegetation pixels)
clean_data2 = dfmndwi.dropna(subset=['MNDWI', 'LST'])

x = clean_data2['MNDWI']
y = clean_data2['LST']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.60, 0.95, text, transform=plt.gca().transAxes,
     fontsize=11, verticalalignment='top')

plt.title('MNDWI vs LST')
plt.xlabel('MNDWI')
plt.ylabel('LST')
plt.figure(figsize=(5, 4))
plt.show()

# LST vs NDVI
clean_data = df3.dropna(subset=['NDVI', 'LST'])

x = clean_data['NDVI']
y = clean_data['LST']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.60, 0.95, text, transform=plt.gca().transAxes,
     fontsize=11, verticalalignment='top')

plt.title('NDVI vs LST')
plt.xlabel('NDVI')
plt.ylabel('LST')
plt.figure(figsize=(5, 4))
plt.show()

# LST vs NDVI (with only vegetation pixels)
clean_data = dfndvi.dropna(subset=['NDVI', 'LST'])

x = clean_data['NDVI']
y = clean_data['LST']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
y_hat = np.poly1d(z)(x)

plt.plot(x, y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y,y_hat):0.3f}$"
plt.gca().text(0.60, 0.95, text, transform=plt.gca().transAxes,
     fontsize=11, verticalalignment='top')

plt.title('NDVI vs LST')
plt.xlabel('NDVI')
plt.ylabel('LST')
plt.figure(figsize=(5, 4))
plt.show()

###################################################################### References for the code ######################################################################

# Carlson, T. N., & Ripley, D. A. (1997). On the relation between NDVI, fractional vegetation cover, and leaf area index. Remote sensing of Environment, 62(3), 241-252.

# Crist, E. P., & Cicone, R. C. (1984). A physically-based transformation of Thematic Mapper data---The TM Tasseled Cap. IEEE Transactions on Geoscience and Remote sensing, (3), 256-263.

# Foga, S., Scaramuzza, P. L., Guo, S., Zhu, Z., Dilley Jr, R. D., Beckmann, T., ... & Laue, B. (2017). Cloud detection algorithm comparison and validation for operational Landsat data products. Remote sensing of environment, 194, 379-390.

# Jimenez-Munoz, J. C., Cristóbal, J., Sobrino, J. A., Sòria, G., Ninyerola, M., & Pons, X. (2008). Revision of the single-channel algorithm for land surface temperature retrieval from Landsat thermal-infrared data. IEEE Transactions on geoscience and remote sensing, 47(1), 339-349.

# Jiménez-Muñoz, J. C., Sobrino, J. A., Skoković, D., Mattar, C., & Cristobal, J. (2014). Land surface temperature retrieval methods from Landsat-8 thermal infrared sensor data. IEEE Geoscience and remote sensing letters, 11(10), 1840-1843.

# Lantz, T. C., Gergel, S. E., & Henry, G. H. (2010). Response of green alder (Alnus viridis subsp. fruticosa) patch dynamics and plant community composition to fire and regional temperature in north‐western Canada. Journal of Biogeography, 37(8), 1597-1610.

# Li, Z. L., Tang, B. H., Wu, H., Ren, H., Yan, G., Wan, Z., ... & Sobrino, J. A. (2013). Satellite-derived land surface temperature: Current status and perspectives. Remote sensing of environment, 131, 14-37.

# Nill, L., Ullmann, T., Kneisel, C., Sobiech-Wolf, J., & Baumhauer, R. (2019). Assessing spatiotemporal variations of Landsat land surface temperature and multispectral indices in the Arctic Mackenzie Delta Region between 1985 and 2018. Remote Sensing, 11(19), 2329.

# Rouse Jr, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1973). Monitoring the vernal advancement and retrogradation (green wave effect) of natural vegetation (No. NASA-CR-132982).

# Sobrino, J. A., Jiménez-Muñoz, J. C., Sòria, G., Romaguera, M., Guanter, L., Moreno, J., ... & Martínez, P. (2008). Land surface emissivity retrieval from different VNIR and TIR sensors. IEEE transactions on geoscience and remote sensing, 46(2), 316-327.

# Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International journal of remote sensing, 27(14), 3025-3033.

# Xu, H. (2008). A new index for delineating built‐up land features in satellite imagery. International journal of remote sensing, 29(14), 4269-4276.