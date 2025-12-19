@echo off

:prepare
set ext=%1
if "%1"=="" set ext=mp4
set dirpath=%~p0
set outpath=%dirpath:~0,-1%.%ext%
set finpath=.\_out.%ext%

if exist %finpath% del %finpath%

:makedir
if exist dir.lst del dir.lst
if exist %outpath% del %outpath%
if exist %finpath% del %finpath%
for %%1 in (*.%ext%) do echo file %%1 >> dir.lst 

:resave
ffmpeg -y -f concat -safe 0 -i dir.lst -map 0 -c copy %outpath%
del dir.lst 

move %outpath% %finpath%

:end
