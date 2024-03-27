#define appName "seg4d"
#define appVersion "1.0"
#define appVerName "SEG4D_Installer"
#define appPublisher ""
#define appUrl ""
#define appExeName "seg4d.py"


[Setup]
AppName={#appName}
AppVersion={#appVersion}
AppVerName={#appVerName}
AppPublisher={#appPublisher}
DefaultDirName={commonpf}\CloudCompare
AllowNoIcons=true
SolidCompression=false
VersionInfoVersion={#appVersion}
VersionInfoCompany={#appPublisher}
VersionInfoProductName={#appName}
VersionInfoProductVersion={#appVersion}
DirExistsWarning=no
AppPublisherURL={#appUrl}
UninstallFilesDir={app}
OutputBaseFilename={#appVerName} Setup
OutputDir=Installer
DefaultGroupName={#appName}
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
DisableStartupPrompt=true
DiskSpanning=yes
SetupIconFile=logo.ico
;Esto es por si se modifica alguna variable de entorno
;ChangesEnvironment=true


[Tasks]
;Name: desktopicon; Description: {cm:CreateDesktopIcon}; GroupDescription: {cm:AdditionalIcons}; Flags: unchecked
;Name: quicklaunchicon; Description: {cm:CreateQuickLaunchIcon}; GroupDescription: {cm:AdditionalIcons}; Flags: unchecked


[Icons]
;De esta forma se añade al menú inicio el ejecutable y el desinstalador
;Name: {group}\{#appName}; Filename: {app}\{#appExeName}; WorkingDir: {app}
;Name: {group}\{#appName}\{cm:UninstallProgram,{#appName}}; Filename: {uninstallexe}
;Name: {commondesktop}\{#appName}; Filename: {app}\{#appExeName}; IconFilename: "{app}\assets\logo.ico"; Tasks: desktopicon; WorkingDir: {app}
;Name: {userappdata}\Microsoft\Internet Explorer\Quick Launch\{#appName}; Filename: {app}\{#appExeName}; Tasks: quicklaunchicon; WorkingDir: {app}

[Files]
;Aqui se seleccionan los ficheros que se quieren instalar. {app} se establece con la ruta de instalación.
Source: "seg4d\seg4d.py"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}"; Tasks: ; Languages: ; Flags: replacesameversion
Source: "seg4d\assets\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\assets"; Flags: recursesubdirs
Source: "seg4d\configs\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\configs"; Flags: recursesubdirs
Source: "seg4d\geometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\geometric-based_methods"; Flags: recursesubdirs
Source: "seg4d\main_module\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\main_module"; Flags: recursesubdirs
Source: "seg4d\other_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\other_methods"; Flags: recursesubdirs
Source: "seg4d\point_clouds_examples\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\point_clouds_examples"; Flags: recursesubdirs
Source: "seg4d\radiometric-based_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\radiometric-based_methods"; Flags: recursesubdirs
Source: "seg4d\segmentation_methods\*"; DestDir: "{app}\plugins\Python\Lib\site-packages\{#appName}\segmentation_methods"; Flags: recursesubdirs

Source: "seg4d_plugin\*"; DestDir: "{app}\plugins-python\seg4d_plugin"; Flags: recursesubdirs

;Se puede copiar los ficheros a un subdirectorio:
;Source: seg4d.py; DestDir: {app}/scripts; Tasks: ; Languages: ; Flags: replacesameversion

;Para seleccionar todos los ficheros de una carpeta:
;Source: files\*.*; DestDir: {app}\files
;si se quiere filtrar por la extensión
;Source: files\*.py; DestDir: {app}\files
;Para copiar todos los archivos y subdirectorios
;Source: files\*; DestDir: {app}\files; Flags: recursesubdirs

[Run]
;Una vez que el instalador ha copiado todos los ficheros al directorio de instalación se instalan las dependencias.
Filename: python; Parameters: -m ensurepip; Flags: runascurrentuser

Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install scipy"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install scikit-learn==1.1.0"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install pandas"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install matplotlib"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install open3d"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install fuzzy-c-means"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install yellowbrick"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install opencv-python"; Flags: runhidden
Filename: "{app}\plugins\Python\python.exe"; Parameters: "-m pip install PyYAML"; Flags: runhidden