
weapon model\animation replacer
standalone python app (old build) but you need to run it and set the paths so maya ui correctly
-first to extract run db files main_ui.py and locate the game files (C:\black\game)


-maya (latest) weapon model\animation\ replacer, direct in maya, also support skinning transfer
-in maya_black_mod_UI.py replace the path (SETTINGS_JSON = r"C:/Users/settings.json")  path ; line 28
(settings.json in the same folder of main_ui.py)
--it's straight forward but soon i'll do tutorial

TEXTURE REPLAECEMENT 
- renderware engine use base color and Alpha maps only it's alpha use for shiny surface like metallic
- a fork of SubCraft that you can load db files and adjust the textures and rebuild db file in the export window
- python app mod_codes/black_texture_tool.py that replace textures if you don't want to use SubCraft

weather strom fx \ tracer \ color grade Loader
Black Loader Based on Spark198rus 
