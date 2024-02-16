###
 # @Author: Chris Xiao yl.xiao@mail.utoronto.ca
 # @Date: 2024-01-12 15:39:29
 # @LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
 # @LastEditTime: 2024-02-16 00:50:09
 # @FilePath: /mbp1413-final/export_env.sh
 # @Description: bash script for exporting conda environment
 # I Love IU
 # Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
### 

# Get pip packages from conda environment
pip_packages=$(conda env export | grep -A9999 ".*- pip:" | grep -v "^prefix: ")

# Export conda environment without builds, and append pip packages
conda env export --from-history | grep -v "^prefix: " > environment.yml
echo "$pip_packages" >> environment.yml