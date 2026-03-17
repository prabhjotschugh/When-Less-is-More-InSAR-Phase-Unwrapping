import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


# --- Data Definition ---
frame_data = {
    'Calatrava': (38.87, -4.02, 'Europe', '#E74C3C', 3.5),
    'Azores': (38.65, -28.0, 'Europe', '#E74C3C', 3.5),
    'Methana': (37.58, 23.37, 'Europe', '#E74C3C', 3.5),
    'Sabalan': (38.25, 47.85, 'Middle East', '#9B59B6', 4.0),
    'Sahand': (37.75, 46.43, 'Middle East', '#9B59B6', 4.0),
    'Middle Gobi': (43.5, 105.0, 'Asia', '#3498DB', 4.0),
    'Sangeang Api': (-8.20, 119.07, 'SE Asia', '#1ABC9C', 5.0),
    'Semeru': (-8.11, 112.92, 'SE Asia', '#1ABC9C', 5.0),
    'Kelut': (-7.93, 112.31, 'SE Asia', '#1ABC9C', 5.0),
    'Java': (-7.70, 112.50, 'SE Asia', '#1ABC9C', 5.0),
    'Black Rock': (40.67, -119.0, 'N America', '#27AE60', 4.5),
    'San Francisco': (35.33, -111.50, 'N America', '#27AE60', 4.5),
    'Durango': (24.0, -104.67, 'N America', '#27AE60', 4.5),
    'Pico de Orizaba': (19.03, -97.27, 'N America', '#27AE60', 4.5),
    'Nunivak': (60.12, -166.3, 'N America', '#27AE60', 4.5),
    'Garibaldi': (49.85, -123.0, 'N America', '#27AE60', 4.5),
    'Hudson/Viedma': (-46.5, -73.0, 'S America', '#16A085', 5.5),
    'Deception': (-62.97, -60.65, 'Antarctica', '#7F8C8D', 6.0),
    'White Island': (-37.52, 177.18, 'Oceania', '#F39C12', 5.5),
    'Okataina': (-38.12, 176.50, 'Oceania', '#F39C12', 5.5),
    'Mayor Island': (-37.28, 176.25, 'Oceania', '#F39C12', 5.5),
}

# --- 1. Setup Figure and Robinson Projection ---
plt.rcParams['font.family'] = 'Calibri'
fig = plt.figure(figsize=(16, 9), facecolor='white')
ax = plt.axes(projection=ccrs.Robinson())

# --- 2. Remove Extra Space ---
# Calculating tight bounds based on data points
lons = [d[1] for d in frame_data.values()]
lats = [d[0] for d in frame_data.values()]
# Note: set_extent works best with PlateCarree even when using Robinson projection
ax.set_extent([min(lons)-15, max(lons)+15, min(lats)-10, max(lats)+10], crs=ccrs.PlateCarree())

# --- 3. Light Theme Styling with Darker Outlines ---
ax.add_feature(cfeature.OCEAN, facecolor='#f0f2f5', zorder=0)
ax.add_feature(cfeature.LAND, facecolor='#e6e9ed', edgecolor='none', zorder=1)

# Darker, crisp continent outlines
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#2c3e50', zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#7f8c8d', alpha=0.6, zorder=2)

# --- 4. Plot Regions and Points with Large Borders ---
for name, (lat, lon, region, color, radius) in frame_data.items():
    # 1. Outer coverage area - Increased alpha from 0.15 to 0.3 for brightness
    ax.add_patch(Circle((lon, lat), radius * 1.5, transform=ccrs.PlateCarree(),
                        facecolor=color, alpha=0.35, zorder=3))
    
    # 2. Inner glow area - Increased alpha from 0.25 to 0.5 for brightness
    ax.add_patch(Circle((lon, lat), radius * 0.6, transform=ccrs.PlateCarree(),
                        facecolor=color, alpha=0.5, zorder=4))
    
    # 3. Main center point - Increased markersize to 12 and markeredgewidth to 3
    ax.plot(lon, lat, marker='o', color=color, markersize=12, 
            markeredgecolor='white', markeredgewidth=3.0,
            transform=ccrs.PlateCarree(), zorder=6)
    
    # 4. Added a dark outer rim to make the bright colors "pop" against light land
    ax.plot(lon, lat, marker='o', color='none', markersize=13, 
            markeredgecolor='#2c3e50', markeredgewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=5)

# --- 5. Legend Construction (Right Side) ---
# Create counts for the legend
regions_order = ['Europe', 'Middle East', 'Asia', 'SE Asia', 'N America', 'S America', 'Oceania', 'Antarctica']
region_colors = {d[2]: d[3] for d in frame_data.values()}
regions_count = {}
for d in frame_data.values():
    regions_count[d[2]] = regions_count.get(d[2], 0) + 1

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f"{reg} ({regions_count[reg]})", 
           markerfacecolor=region_colors[reg], markersize=12, 
           markeredgecolor='#2c3e50', markeredgewidth=1)
    for reg in regions_order if reg in region_colors
]

# Style and position the legend on the right
leg = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                title="Regions", title_fontsize=13, fontsize=12,
                frameon=True, fancybox=False, edgecolor='#2c3e50', facecolor='white')
leg.get_frame().set_linewidth(0.8)

# --- 6. Final Polish ---
plt.tight_layout()
plt.savefig('map.png', dpi=300, bbox_inches='tight')
plt.show()