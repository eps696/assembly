@echo off

call ../env.bat

set vis=-imod "bfl:5@1" -isz 1344-752 -vmod "bytedance:2@2" -vsz 864-480 -fps 24 -iref 8
set args=-qa 3 -np 4 

REM set mode=author
set mode=chat

set llm=-a lms -tmod openai/gpt-oss-20b -lh white
REM set llm=-a lms -tmod gpt-4o-mini
REM set llm=-a adk -tmod gemini-2.5-flash 

python src/%mode%.py %args% %llm% -v  -txt
REM python src/%mode%.py %args% %llm% -v  %vis% 
REM python src/%mode%.py -json _out/log.json -arg _out/config.txt 
