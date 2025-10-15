function varargout = NM_OOCV_Manager(varargin)
% NM_OOCV_MANAGER MATLAB code for NM_OOCV_Manager.fig
%      NM_OOCV_MANAGER by itself, creates a new NM_OOCV_MANAGER or raises the
%      existing singleton*.
%
%      H = NM_OOCV_MANAGER returns the handle to a new NM_OOCV_MANAGER or the handle to
%      the existing singleton*.
%
%      NM_OOCV_MANAGER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NM_OOCV_MANAGER.M with the given input arguments.
%
%      NM_OOCV_MANAGER('Property','Value',...) creates a new NM_OOCV_MANAGER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NM_OOCV_Manager_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NM_OOCV_Manager_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NM_OOCV_Manager

% Last Modified by GUIDE v2.5 26-May-2025 11:40:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @NM_OOCV_Manager_OpeningFcn, ...
                   'gui_OutputFcn',  @NM_OOCV_Manager_OutputFcn, ...
                   'gui_LayoutFcn',  @NM_OOCV_Manager_LayoutFcn, ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before NM_OOCV_Manager is made visible.
function NM_OOCV_Manager_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to NM_OOCV_Manager (see VARARGIN)

% Choose default command line output for NM_OOCV_Manager
handles.output = [];

% Update handles structure
guidata(hObject, handles);

% Insert custom Title and Text if specified by the user
% Hint: when choosing keywords, be sure they are not easily confused 
% with existing figure properties.  See the output of set(figure) for
% a list of figure properties.
if(nargin > 3)
    for index = 1:2:(nargin-3)
        if nargin-3==index, break, end
        switch lower(varargin{index})
             case 'title'
                set(hObject, 'Name', varargin{index+1});
             case 'list'
                 if ~isempty(varargin{index+1})
                    for i=1:numel(varargin{index+1})
                        handles.Data.Items{i} = varargin{index+1}{i};
                    end
                 else
                    handles.Data.Items=[];
                 end
                 guidata(handles.figure1,handles);
                 print_dataitems(handles)
              case 'mode'
                 switch varargin{index+1}
                     case 'select'
                         bt_str = 'Select dataset'; vis_str = 'off';
                         set(handles.lstData,'Position',[0.034 0.05 0.93 0.82]);
                         handles.mode = 0;
                     case 'modify'
                         bt_str = 'Add/Modify Data in Dataset'; vis_str = 'on';
                         set(handles.lstData,'Position',[0.034 0.17 0.93 0.70]);
                         handles.mode = 1;
                 end
                 set(handles.cmdSelect,'String',bt_str);
                 set(handles.cmdDelete,'Visible',vis_str);
                 set(handles.txtNewData,'Visible',vis_str);
                 set(handles.cmdCreateNew,'Visible',vis_str);
                 set(handles.chkLabelKnown,'Visible',vis_str);
                 set(handles.chkCalibrationDataAvail,'Visible',vis_str);
                 set(handles.cmdSave,'Visible',vis_str);
            case 'multiselection'
                switch varargin{index+1}
                    case 0
                        set(handles.lstData,'Max', 1,'Min',0);
                    case 1
                        set(handles.lstData,'Max', numel(handles.Data.Items),'Min',0);
                end
        end
    end
end

% Determine the position of the dialog - centered on the callback figure
% if available, else, centered on the screen
% Get default figure position
FigPos = get(0, 'DefaultFigurePosition');

% Store original units and position
OldUnits = get(hObject, 'Units');
set(hObject, 'Units', 'pixels');
OldPos = get(hObject, 'Position');

% Get screen size in pixels
if isempty(gcbf)
    ScreenUnits = get(0, 'Units');
    set(0, 'Units', 'pixels');
    ScreenSize = get(0, 'ScreenSize');  % [left bottom width height]
    set(0, 'Units', ScreenUnits);
    
    % Desired width and height: 2/3 of width, 1/3 of height
    FigWidth = round(2/3 * ScreenSize(3));
    FigHeight = round(1/3 * ScreenSize(4));
    
    % Center horizontally and place vertically around 2/3 up the screen
    FigPos(1) = (ScreenSize(3) - FigWidth) / 2;
    FigPos(2) = (ScreenSize(4) - FigHeight) / 2;
else
    GCBFOldUnits = get(gcbf, 'Units');
    set(gcbf, 'Units', 'pixels');
    GCBFPos = get(gcbf, 'Position');
    set(gcbf, 'Units', GCBFOldUnits);
    
    FigWidth = round(2/3 * GCBFPos(3));
    FigHeight = round(1/3 * GCBFPos(4));
    
    FigPos(1:2) = [(GCBFPos(1) + GCBFPos(3)/2 - FigWidth/2), ...
                   (GCBFPos(2) + GCBFPos(4)/2 - FigHeight/2)];
end

% Set width and height
FigPos(3:4) = [FigWidth, FigHeight];

% Apply the new position
set(hObject, 'Position', FigPos);
set(hObject, 'Units', OldUnits);

% Make the GUI modal
set(handles.figure1,'WindowStyle','modal')

% UIWAIT makes NM_OOCV_Manager wait for user response (see UIRESUME)
uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = NM_OOCV_Manager_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% The figure can be deleted now
delete(handles.figure1);

% --- Executes on button press in cmdSelect.
function cmdSelect_Callback(hObject, eventdata, handles)
% hObject    handle to cmdSelect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.Data.SelItem = get(handles.lstData,'Value');
handles.output = handles.Data;
handles.output.exit = true;
% Update handles structure
guidata(hObject, handles);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.output = handles.Data;
handles.output.exit = true;
% Update handles structure
guidata(hObject, handles);

% Use UIRESUME instead of delete because the OutputFcn needs
% to get the updated handles structure.
uiresume(handles.figure1);


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end
handles.output = handles.Data;
handles.output.exit = true;
% Update handles structure
guidata(hObject, handles);

% --- Executes on key press over figure1 with no controls selected.
function figure1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check for "enter" or "escape"
if isequal(get(hObject,'CurrentKey'),'escape')
    % User said no by hitting escape
    handles.output = 'No';
    
    % Update handles structure
    guidata(hObject, handles);
    
    uiresume(handles.figure1);
end    
    
if isequal(get(hObject,'CurrentKey'),'return')
    uiresume(handles.figure1);
end    


% --- Executes on selection change in lstData.
function lstData_Callback(hObject, eventdata, handles)
% hObject    handle to lstData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns lstData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lstData


% --- Executes during object creation, after setting all properties.
function lstData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lstData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function txtNewData_Callback(hObject, eventdata, handles)
% hObject    handle to txtNewData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtNewData as text
%        str2double(get(hObject,'String')) returns contents of txtNewData as a double


% --- Executes during object creation, after setting all properties.
function txtNewData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtNewData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cmdCreateNew.
function cmdCreateNew_Callback(hObject, eventdata, handles)
% hObject    handle to cmdCreateNew (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isfield(handles.Data,'Items'), l = numel(handles.Data.Items)+1; else l=1; end
handles.Data.Items{l} = struct('desc', get(handles.txtNewData,'String'), ...
                                'date',datestr(now), ...
                                'n_subjects_all',0, ...
                                'labels_known', get(handles.chkLabelKnown,'Value'), ...
                                'os', sprintf('%s', system_dependent('getos')));
if isfield(handles.Data,'NewItemIndex'), lx = numel(handles.Data.NewItemIndex)+1; else lx = 1; end
handles.Data.NewItemIndex(lx) = l;
print_dataitems(handles)
guidata(handles.figure1,handles);

% --- Executes on button press in cmdDelete.
function cmdDelete_Callback(hObject, eventdata, handles)
% hObject    handle to cmdDelete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

selItem = get(handles.lstData,'Value');
lstItems = get(handles.lstData,'String');
delfl = questdlg(sprintf('Are you sure you want to delete dataset:\n%s',lstItems{selItem}));
if strcmp(delfl,'Yes')
%     handles.Data.Items(selItem) = [];
%     if isfield(handles.Data,'NewItems')
%         ind = find(handles.Data.NewItems, selItem);
%         handles.Data.NewItems(ind)=[];
%     end

    handles.output.delete = selItem;
    guidata(handles.figure1,handles);
    uiresume(handles.figure1)

%     % Update handles structure
%     guidata(hObject, handles);
%     
%     print_dataitems(handles,selItem-1);
end

function toggle_controls(handles)

if ~isfield(handles,'Data') || ~isfield(handles.Data,'Items') || isempty(handles.Data.Items);
    tglstr = 'off';
else
    tglstr = 'on';
end

set(handles.cmdSelect,'Enable',tglstr);
set(handles.cmdSave,'Enable',tglstr);
set(handles.cmdDelete,'Enable',tglstr);
set(handles.lstData,'Enable',tglstr);

function print_dataitems(handles, value)
if ~exist('value','var') || isempty(value), value = numel(handles.Data.Items); end
if ~isempty(handles.Data.Items)
    for i=1:numel(handles.Data.Items)
        if isfield(handles.Data.Items{i},'labels_known')
            if handles.Data.Items{i}.labels_known
                lbknown_str = '; labels known: yes';
            else
                lbknown_str = '; labels known: no';
            end
        else
            lbknown_str = '';
        end
        if ~isfield(handles.Data.Items{i},'n_subjects_all')
            n_subjects_all=0;
        else
            n_subjects_all= handles.Data.Items{i}.n_subjects_all;
        end
        datastr{i} = sprintf('[%3i] %s (%g cases); date: %s%s',i,handles.Data.Items{i}.desc,n_subjects_all,handles.Data.Items{i}.date, lbknown_str);
    end
    set(handles.lstData, 'String', datastr);
    handles.lstData.Value=value;
else
    set(handles.lstData, 'String', 'NO DATA AVALABLE');
    handles.lstData.Value=1;
end
 toggle_controls(handles)

% --- Executes on button press in cmdSave.
function cmdSave_Callback(hObject, eventdata, handles)
% hObject    handle to cmdSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

selItem = get(handles.lstData,'Value');
%lstItems = get(handles.lstData,'String');
OOCV = handles.Data.Items{selItem};
filename = 'NM_OOCV_';
uisave('OOCV',filename);

% --- Executes on button press in chkLabelKnown.
function chkLabelKnown_Callback(hObject, eventdata, handles)
% hObject    handle to chkLabelKnown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chkLabelKnown


% --- Executes on button press in chkCalibrationDataAvail.
function chkCalibrationDataAvail_Callback(hObject, eventdata, handles)
% hObject    handle to chkCalibrationDataAvail (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of chkCalibrationDataAvail


% --------------------------------------------------------------------
function uiLoadData_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uiLoadData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function uiNewData_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uiNewData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in cmdLoad.
function cmdLoad_Callback(hObject, eventdata, handles)
% hObject    handle to cmdLoad (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

n = numel(handles.Data.Items);

[filename,pathname] = uigetfile('*.mat','Load Independent Test Data','MultiSelect','on');
if iscell(filename)
    nF = numel(filename);
else
    nF=1;
end
if filename 
    for i=1:nF
        if iscell(filename)
            pth = fullfile(pathname,filename{i});
            load(pth)
        else
            pth = fullfile(pathname,filename);
            load(pth)
        end
        handles.Data.Items{n+i} = OOCV;
    end
end
handles.Data.load=true;
guidata(handles.figure1,handles);
print_dataitems(handles)


% --- Creates and returns a handle to the GUI figure. 
function h1 = NM_OOCV_Manager_LayoutFcn(policy)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

persistent hsingleton;
if strcmpi(policy, 'reuse') & ishandle(hsingleton)
    h1 = hsingleton;
    return;
end
load NM_OOCV_Manager.mat


appdata = [];
appdata.GUIDEOptions = struct(...
    'active_h', [], ...
    'taginfo', struct(...
    'figure', 2, ...
    'pushbutton', 8, ...
    'axes', 2, ...
    'text', 2, ...
    'listbox', 2, ...
    'edit', 2, ...
    'checkbox', 3, ...
    'uitoolbar', 2, ...
    'uipushtool', 5, ...
    'uitoggletool', 2), ...
    'override', 1, ...
    'release', 13, ...
    'resize', 'none', ...
    'accessibility', 'callback', ...
    'mfile', 1, ...
    'callbacks', 1, ...
    'singleton', 1, ...
    'syscolorfig', 1, ...
    'blocking', 0, ...
    'lastSavedFile', 'C:\Users\nikol\OneDrive - LMU Klinikum AöR\Software\NeuroMiner_Current\gui\NM_OOCV_Manager.m', ...
    'lastFilename', 'D:\NeuroMiner_Elessar\gui\NM_OOCV_Manager.fig');
appdata.lastValidTag = 'figure1';
appdata.GUIDELayoutEditor = mat{1};

h1 = figure(...
'PaperUnits',get(0,'defaultfigurePaperUnits'),...
'Units','characters',...
'Position',[131.214285714286 15.5294117647059 109.214285714286 31.3823529411765],...
'Visible',get(0,'defaultfigureVisible'),...
'Color',get(0,'defaultfigureColor'),...
'CloseRequestFcn',@(hObject,eventdata)NM_OOCV_Manager('figure1_CloseRequestFcn',hObject,eventdata,guidata(hObject)),...
'IntegerHandle','off',...
'Colormap',[0 0 0;1 1 1;0.984313725490196 0.956862745098039 0.6;0.984313725490196 0.952941176470588 0.6;0 0 0.6;0.988235294117647 0.956862745098039 0.603921568627451;0.988235294117647 0.956862745098039 0.6;0.690196078431373 0.662745098039216 0.666666666666667;0.0549019607843137 0.0509803921568627 0.0549019607843137;0.0627450980392157 0.0588235294117647 0.0627450980392157;0.0705882352941176 0.0666666666666667 0.0705882352941176;0.0156862745098039 0.0117647058823529 0.0196078431372549;0.0352941176470588 0.0313725490196078 0.0392156862745098;0.623529411764706 0.596078431372549 0.658823529411765;0.0196078431372549 0.0156862745098039 0.0274509803921569;0.501960784313725 0.482352941176471 0.647058823529412;0.447058823529412 0.427450980392157 0.643137254901961;0.388235294117647 0.372549019607843 0.63921568627451;0.270588235294118 0.258823529411765 0.627450980392157;0.294117647058824 0.282352941176471 0.627450980392157;0.309803921568627 0.298039215686275 0.631372549019608;0.352941176470588 0.341176470588235 0.635294117647059;0.125490196078431 0.12156862745098 0.611764705882353;0.149019607843137 0.145098039215686 0.615686274509804;0.192156862745098 0.184313725490196 0.619607843137255;0.223529411764706 0.215686274509804 0.619607843137255;0 0 0.00784313725490196;0 0 0.00392156862745098;0.0196078431372549 0.0196078431372549 0.603921568627451;0.0470588235294118 0.0431372549019608 0.603921568627451;0.0784313725490196 0.0745098039215686 0.607843137254902;0.0823529411764706 0.0784313725490196 0.607843137254902;0.105882352941176 0.101960784313725 0.611764705882353;0.00392156862745098 0.00392156862745098 0.0196078431372549;0.00784313725490196 0.00784313725490196 0.0196078431372549;0.0117647058823529 0.0117647058823529 0.0274509803921569;0.0235294117647059 0.0235294117647059 0.0352941176470588;0.0274509803921569 0.0274509803921569 0.0392156862745098;1 0.996078431372549 0.623529411764706;1 1 0.627450980392157;1 0.996078431372549 0.631372549019608;1 1 0.635294117647059;1 1 0.643137254901961;1 1 0.650980392156863;0.0705882352941176 0.0705882352941176 0.0509803921568627;0.305882352941176 0.305882352941176 0.227450980392157;0.16078431372549 0.16078431372549 0.12156862745098;0.0392156862745098 0.0392156862745098 0.0352941176470588;0.0705882352941176 0.0705882352941176 0.0666666666666667;0.0862745098039216 0.0862745098039216 0.0823529411764706;0.184313725490196 0.184313725490196 0.176470588235294;0.0941176470588235 0.0941176470588235 0.0901960784313725;0.101960784313725 0.101960784313725 0.0980392156862745;0.145098039215686 0.145098039215686 0.141176470588235;1 0.988235294117647 0.615686274509804;1 0.992156862745098 0.619607843137255;0.925490196078431 0.913725490196078 0.6;0.423529411764706 0.419607843137255 0.298039215686275;1 0.976470588235294 0.611764705882353;0.996078431372549 0.972549019607843 0.607843137254902;1 0.980392156862745 0.615686274509804;1 0.984313725490196 0.619607843137255;1 0.976470588235294 0.619607843137255;0.988235294117647 0.972549019607843 0.615686274509804;1 0.980392156862745 0.627450980392157;0.988235294117647 0.972549019607843 0.619607843137255;0.984313725490196 0.964705882352941 0.615686274509804;0.219607843137255 0.215686274509804 0.145098039215686;0.4 0.392156862745098 0.270588235294118;0.258823529411765 0.254901960784314 0.192156862745098;0.145098039215686 0.141176470588235 0.0862745098039216;0.992156862745098 0.96078431372549 0.603921568627451;0.988235294117647 0.96078431372549 0.6;0.96078431372549 0.929411764705882 0.584313725490196;0.996078431372549 0.968627450980392 0.607843137254902;0.988235294117647 0.96078431372549 0.603921568627451;0.96078431372549 0.933333333333333 0.588235294117647;0.945098039215686 0.913725490196078 0.576470588235294;0.996078431372549 0.964705882352941 0.611764705882353;0.984313725490196 0.952941176470588 0.603921568627451;0.964705882352941 0.941176470588235 0.592156862745098;0.964705882352941 0.937254901960784 0.592156862745098;0.956862745098039 0.925490196078431 0.588235294117647;0.949019607843137 0.92156862745098 0.584313725490196;0.984313725490196 0.96078431372549 0.607843137254902;0.952941176470588 0.925490196078431 0.588235294117647;0.972549019607843 0.949019607843137 0.607843137254902;0.956862745098039 0.929411764705882 0.6;0.937254901960784 0.909803921568627 0.588235294117647;0.929411764705882 0.901960784313726 0.584313725490196;0.92156862745098 0.898039215686275 0.584313725490196;0.909803921568627 0.882352941176471 0.576470588235294;0.850980392156863 0.827450980392157 0.541176470588235;0.611764705882353 0.596078431372549 0.4;0.407843137254902 0.396078431372549 0.270588235294118;0.458823529411765 0.447058823529412 0.309803921568627;0.368627450980392 0.36078431372549 0.258823529411765;0.329411764705882 0.32156862745098 0.235294117647059;0.231372549019608 0.227450980392157 0.176470588235294;0.988235294117647 0.952941176470588 0.6;0.988235294117647 0.952941176470588 0.603921568627451;0.984313725490196 0.949019607843137 0.6;0.92156862745098 0.890196078431372 0.580392156862745;0.819607843137255 0.792156862745098 0.52156862745098;0.83921568627451 0.811764705882353 0.537254901960784;0.8 0.772549019607843 0.509803921568627;0.764705882352941 0.737254901960784 0.494117647058824;0.713725490196078 0.690196078431373 0.462745098039216;0.741176470588235 0.713725490196078 0.482352941176471;0.580392156862745 0.56078431372549 0.380392156862745;0.215686274509804 0.207843137254902 0.141176470588235;0.698039215686274 0.674509803921569 0.458823529411765;0.619607843137255 0.6 0.407843137254902;0.682352941176471 0.658823529411765 0.450980392156863;0.450980392156863 0.435294117647059 0.301960784313725;0.262745098039216 0.254901960784314 0.176470588235294;0.584313725490196 0.564705882352941 0.396078431372549;0.486274509803922 0.470588235294118 0.329411764705882;0.6 0.580392156862745 0.407843137254902;0.470588235294118 0.454901960784314 0.32156862745098;0.505882352941176 0.490196078431373 0.349019607843137;0.388235294117647 0.376470588235294 0.274509803921569;0.403921568627451 0.392156862745098 0.290196078431373;0.266666666666667 0.258823529411765 0.192156862745098;0.180392156862745 0.176470588235294 0.137254901960784;0.72156862745098 0.694117647058824 0.470588235294118;0.6 0.576470588235294 0.392156862745098;0.101960784313725 0.0980392156862745 0.0705882352941176;0.309803921568627 0.298039215686275 0.215686274509804;0.313725490196078 0.301960784313725 0.219607843137255;0.250980392156863 0.243137254901961 0.180392156862745;0.141176470588235 0.137254901960784 0.105882352941176;0.156862745098039 0.152941176470588 0.12156862745098;0.0862745098039216 0.0823529411764706 0.0588235294117647;0.494117647058824 0.474509803921569 0.349019607843137;0.286274509803922 0.274509803921569 0.203921568627451;0.219607843137255 0.211764705882353 0.164705882352941;0.243137254901961 0.235294117647059 0.184313725490196;0.0627450980392157 0.0588235294117647 0.0392156862745098;0.192156862745098 0.184313725490196 0.145098039215686;0.443137254901961 0.43921568627451 0.419607843137255;0.0784313725490196 0.0745098039215686 0.0588235294117647;0.164705882352941 0.156862745098039 0.125490196078431;0.117647058823529 0.113725490196078 0.0980392156862745;0.152941176470588 0.145098039215686 0.117647058823529;0.850980392156863 0.815686274509804 0.682352941176471;0.835294117647059 0.8 0.67843137254902;0.0470588235294118 0.0431372549019608 0.0313725490196078;0.0862745098039216 0.0823529411764706 0.0705882352941176;0.803921568627451 0.772549019607843 0.67843137254902;0.23921568627451 0.235294117647059 0.223529411764706;0.513725490196078 0.505882352941176 0.482352941176471;0.568627450980392 0.56078431372549 0.537254901960784;0.56078431372549 0.552941176470588 0.529411764705882;0.556862745098039 0.549019607843137 0.525490196078431;0.552941176470588 0.545098039215686 0.52156862745098;0.270588235294118 0.266666666666667 0.254901960784314;0.607843137254902 0.6 0.576470588235294;0.576470588235294 0.568627450980392 0.545098039215686;0.290196078431373 0.286274509803922 0.274509803921569;0.498039215686275 0.490196078431373 0.470588235294118;0.482352941176471 0.474509803921569 0.454901960784314;0.47843137254902 0.470588235294118 0.450980392156863;0.533333333333333 0.525490196078431 0.505882352941176;0.529411764705882 0.52156862745098 0.501960784313725;0.513725490196078 0.505882352941176 0.486274509803922;0.505882352941176 0.498039215686275 0.47843137254902;0.501960784313725 0.494117647058824 0.474509803921569;0.552941176470588 0.545098039215686 0.525490196078431;0.772549019607843 0.741176470588235 0.674509803921569;0.662745098039216 0.650980392156863 0.623529411764706;0.647058823529412 0.635294117647059 0.607843137254902;0.701960784313725 0.690196078431373 0.662745098039216;0.686274509803922 0.674509803921569 0.647058823529412;0.670588235294118 0.658823529411765 0.631372549019608;0.0352941176470588 0.0313725490196078 0.0235294117647059;0.129411764705882 0.12156862745098 0.105882352941176;0.6 0.588235294117647 0.564705882352941;0.588235294117647 0.576470588235294 0.552941176470588;0.580392156862745 0.568627450980392 0.545098039215686;0.63921568627451 0.627450980392157 0.603921568627451;0.627450980392157 0.615686274509804 0.592156862745098;0.623529411764706 0.611764705882353 0.588235294117647;0.619607843137255 0.607843137254902 0.584313725490196;0.611764705882353 0.6 0.576470588235294;0.423529411764706 0.415686274509804 0.4;0.686274509803922 0.674509803921569 0.650980392156863;0.682352941176471 0.670588235294118 0.647058823529412;0.67843137254902 0.666666666666667 0.643137254901961;0.674509803921569 0.662745098039216 0.63921568627451;0.666666666666667 0.654901960784314 0.631372549019608;0.662745098039216 0.650980392156863 0.627450980392157;0.654901960784314 0.643137254901961 0.619607843137255;0.650980392156863 0.63921568627451 0.615686274509804;0.454901960784314 0.447058823529412 0.431372549019608;0.450980392156863 0.443137254901961 0.427450980392157;0.43921568627451 0.431372549019608 0.415686274509804;0.435294117647059 0.427450980392157 0.411764705882353;0.227450980392157 0.223529411764706 0.215686274509804;0.466666666666667 0.458823529411765 0.443137254901961;0.247058823529412 0.243137254901961 0.235294117647059;0.243137254901961 0.23921568627451 0.231372549019608;0.23921568627451 0.235294117647059 0.227450980392157;0.235294117647059 0.231372549019608 0.223529411764706;0.258823529411765 0.254901960784314 0.247058823529412;0.294117647058824 0.290196078431373 0.282352941176471;0.32156862745098 0.317647058823529 0.309803921568627;0.745098039215686 0.717647058823529 0.670588235294118;0.529411764705882 0.517647058823529 0.498039215686275;0.52156862745098 0.509803921568627 0.490196078431373;0.572549019607843 0.56078431372549 0.541176470588235;0.564705882352941 0.552941176470588 0.533333333333333;0.545098039215686 0.533333333333333 0.513725490196078;0.592156862745098 0.580392156862745 0.56078431372549;0.694117647058824 0.67843137254902 0.654901960784314;0.352941176470588 0.345098039215686 0.333333333333333;0.345098039215686 0.337254901960784 0.325490196078431;0.368627450980392 0.36078431372549 0.349019607843137;0.407843137254902 0.4 0.388235294117647;0.4 0.392156862745098 0.380392156862745;0.388235294117647 0.380392156862745 0.368627450980392;0.490196078431373 0.47843137254902 0.462745098039216;0.474509803921569 0.462745098039216 0.447058823529412;0.733333333333333 0.705882352941177 0.670588235294118;0.0588235294117647 0.0549019607843137 0.0509803921568627;0.101960784313725 0.0980392156862745 0.0941176470588235;0.133333333333333 0.129411764705882 0.125490196078431;0.701960784313725 0.682352941176471 0.662745098039216;0.27843137254902 0.270588235294118 0.262745098039216;0.145098039215686 0.141176470588235 0.137254901960784;0.333333333333333 0.325490196078431 0.317647058823529;0.317647058823529 0.309803921568627 0.301960784313725;0.309803921568627 0.301960784313725 0.294117647058824;0.164705882352941 0.16078431372549 0.156862745098039;0.203921568627451 0.2 0.196078431372549;0.0196078431372549 0.0156862745098039 0.0156862745098039;0.0470588235294118 0.0431372549019608 0.0431372549019608;0.0549019607843137 0.0509803921568627 0.0509803921568627;0.0705882352941176 0.0666666666666667 0.0666666666666667;0.0745098039215686 0.0705882352941176 0.0705882352941176;0.0784313725490196 0.0745098039215686 0.0745098039215686;0.12156862745098 0.117647058823529 0.117647058823529;0.113725490196078 0.109803921568627 0.109803921568627;0.172549019607843 0.168627450980392 0.168627450980392;0.109803921568627 0.109803921568627 0.109803921568627;0.105882352941176 0.105882352941176 0.105882352941176;0.0941176470588235 0.0941176470588235 0.0941176470588235;0.0823529411764706 0.0823529411764706 0.0823529411764706;0.0627450980392157 0.0627450980392157 0.0627450980392157;0.0588235294117647 0.0588235294117647 0.0588235294117647;0.0509803921568627 0.0509803921568627 0.0509803921568627;0.0392156862745098 0.0392156862745098 0.0392156862745098;0.0313725490196078 0.0313725490196078 0.0313725490196078;0.0274509803921569 0.0274509803921569 0.0274509803921569;0.00784313725490196 0.00784313725490196 0.00784313725490196;0.752941176470588 0.752941176470588 0.752941176470588],...
'MenuBar','none',...
'ToolBar','none',...
'Name','NM_OOCV_Manager',...
'NumberTitle','off',...
'HandleVisibility','callback',...
'Tag','figure1',...
'UserData',[],...
'Resize','off',...
'PaperPosition',get(0,'defaultfigurePaperPosition'),...
'PaperSize',[20.99999864 29.69999902],...
'PaperType',get(0,'defaultfigurePaperType'),...
'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
'ScreenPixelsPerInchMode','manual',...
'KeyPressFcn',@(hObject,eventdata)NM_OOCV_Manager('figure1_KeyPressFcn',hObject,eventdata,guidata(hObject)),...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'cmdSelect';

h2 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Select Dataset',...
'Position',[0.0321428571428571 0.896907216494845 0.323214285714286 0.0747422680412371],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('cmdSelect_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tag','cmdSelect',...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'pushbutton2';

h3 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Exit',...
'Position',[0.785714285714286 0.899484536082474 0.171428571428571 0.0721649484536082],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('pushbutton2_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Exit Independent Test Data Manager',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Exit Independent Test Data Manager',...
'Tag','pushbutton2',...
'UserData',[],...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'lstData';

h4 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'HorizontalAlignment','left',...
'String',{  'LMU dataset'; 'UBS dataset'; 'UKK dataset' },...
'Style','listbox',...
'Value',1,...
'Position',[0.0347985347985348 0.223039215686274 0.917582417582418 0.63],...
'BackgroundColor',[0.803921568627451 0.87843137254902 0.968627450980392],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('lstData_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Select analyses to operate on',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Select analyses to operate on',...
'CreateFcn', {@local_CreateFcn, @(hObject,eventdata)NM_OOCV_Manager('lstData_CreateFcn',hObject,eventdata,guidata(hObject)), appdata} ,...
'Tag','lstData',...
'FontSize',9);

appdata = [];
appdata.lastValidTag = 'txtNewData';

h5 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'HorizontalAlignment','left',...
'String',blanks(0),...
'Style','edit',...
'Position',[0.0311355311355311 0.0588235294117647 0.40 0.0686274509803921],...
'BackgroundColor',[0.8 1 1],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('txtNewData_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Enter new dataset description here',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Enter new dataset description here',...
'CreateFcn', {@local_CreateFcn, @(hObject,eventdata)NM_OOCV_Manager('txtNewData_CreateFcn',hObject,eventdata,guidata(hObject)), appdata} ,...
'Tag','txtNewData',...
'FontSize',10);

appdata = [];
appdata.lastValidTag = 'cmdDelete';

h6 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Delete',...
'Position',[0.585714285714286 0.896907216494845 0.110714285714286 0.0747422680412371],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('cmdDelete_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Delete selected independent test data',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Delete selected independent test data',...
'Tag','cmdDelete',...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'cmdLoad';

h7 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Load',...
'Position',[0.371428571428571 0.896907216494845 0.1 0.0747422680412371],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('cmdLoad_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Load independent test data',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Load independent test data',...
'Tag','cmdLoad',...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'cmdSave';

h8 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Save',...
'Position',[0.478571428571429 0.896907216494845 0.0999999999999999 0.0747422680412371],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('cmdSave_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tooltip','Save selected independent test data',...
'TooltipMode',get(0,'defaultuicontrolTooltipMode'),...
'TooltipString','Save selected independent test data',...
'Tag','cmdSave',...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'chkLabelKnown';

h9 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'String','Target labels known',...
'Style','checkbox',...
'Position',[26 -0.2 25.8 1.76923076923077],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('chkLabelKnown_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tag','chkLabelKnown',...
'UserData',[],...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'chkCalibrationDataAvail';

h10 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'String','Calibration data available',...
'Style','checkbox',...
'Position',[33 -0.2 30.4 1.76923076923077],...
'BackgroundColor',[0.941176470588235 0.941176470588235 0.941176470588235],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('chkCalibrationDataAvail_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Enable','off',...
'Tag','chkCalibrationDataAvail',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'cmdCreateNew';

h11 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'FontUnits',get(0,'defaultuicontrolFontUnits'),...
'ListboxTop',0,...
'String','Create New ...',...
'Position',[0.78021978021978 0.0392156862745098 0.172161172161172 0.100490196078431],...
'BackgroundColor',[0.67843137254902 0.92156862745098 1],...
'Callback',@(hObject,eventdata)NM_OOCV_Manager('cmdCreateNew_Callback',hObject,eventdata,guidata(hObject)),...
'Children',[],...
'Tag','cmdCreateNew',...
'FontWeight','bold',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.lastValidTag = 'uiIndepToolbar';

h12 = uitoolbar(...
'Parent',h1,...
'Visible','off',...
'Tag','uiIndepToolbar',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.toolid = 'Standard.NewFigure';
appdata.CallbackInUse = struct(...
    'ClickedCallback', 'NM_OOCV_Manager(''uiNewData_ClickedCallback'',gcbo,[],guidata(gcbo))');
appdata.lastValidTag = 'uiNewData';

h13 = uipushtool(...
'Parent',h12,...
'Children',[],...
'Tag','uiNewData',...
'Tooltip','Create new independent test data item',...
'TooltipMode',get(0,'defaultuipushtoolTooltipMode'),...
'CData',mat{2},...
'ClickedCallback',@(hObject,eventdata)NM_OOCV_Manager('uiNewData_ClickedCallback',hObject,eventdata,guidata(hObject)),...
'TooltipString','Create new independent test data item',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.toolid = 'Standard.NewFigure';
appdata.CallbackInUse = struct(...
    'ClickedCallback', '%default');
appdata.lastValidTag = 'uiImportData';

h14 = uipushtool(...
'Parent',h12,...
'Children',[],...
'Tag','uiImportData',...
'Tooltip','Import Data to selected item',...
'TooltipMode',get(0,'defaultuipushtoolTooltipMode'),...
'CData',mat{3},...
'ClickedCallback','%default',...
'TooltipString','Import Data to selected item',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.toolid = 'Standard.FileOpen';
appdata.CallbackInUse = struct(...
    'ClickedCallback', 'NM_OOCV_Manager(''uiLoadData_ClickedCallback'',gcbo,[],guidata(gcbo))');
appdata.lastValidTag = 'uiLoadData';

h15 = uipushtool(...
'Parent',h12,...
'Children',[],...
'BusyAction','cancel',...
'Interruptible','off',...
'Tag','uiLoadData',...
'Tooltip','Open independent test data',...
'TooltipMode',get(0,'defaultuipushtoolTooltipMode'),...
'CData',mat{4},...
'ClickedCallback',@(hObject,eventdata)NM_OOCV_Manager('uiLoadData_ClickedCallback',hObject,eventdata,guidata(hObject)),...
'TooltipString','Open independent test data',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.toolid = 'Standard.SaveFigure';
appdata.CallbackInUse = struct(...
    'ClickedCallback', '%default');
appdata.lastValidTag = 'uiSaveData';

h16 = uipushtool(...
'Parent',h12,...
'Children',[],...
'BusyAction','cancel',...
'Interruptible','off',...
'Tag','uiSaveData',...
'Tooltip','Save Figure',...
'TooltipMode',get(0,'defaultuipushtoolTooltipMode'),...
'CData',mat{5},...
'ClickedCallback','%default',...
'TooltipString','Save Figure',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );

appdata = [];
appdata.toolid = 'Exploration.ZoomIn';
appdata.CallbackInUse = struct(...
    'ClickedCallback', '%default');
appdata.lastValidTag = 'uiInspectPredictions';

h17 = uitoggletool(...
'Parent',h12,...
'Children',[],...
'Tag','uiInspectPredictions',...
'Tooltip','Inspect findings',...
'TooltipMode',get(0,'defaultuitoggletoolTooltipMode'),...
'CData',mat{6},...
'ClickedCallback','%default',...
'TooltipString','Inspect findings',...
'CreateFcn', {@local_CreateFcn, blanks(0), appdata} );


hsingleton = h1;


% --- Set application data first then calling the CreateFcn. 
function local_CreateFcn(hObject, eventdata, createfcn, appdata)

if ~isempty(appdata)
   names = fieldnames(appdata);
   for i=1:length(names)
       name = char(names(i));
       setappdata(hObject, name, getfield(appdata,name));
   end
end

if ~isempty(createfcn)
   if isa(createfcn,'function_handle')
       createfcn(hObject, eventdata);
   else
       eval(createfcn);
   end
end


% --- Handles default GUIDE GUI creation and callback dispatch
function varargout = gui_mainfcn(gui_State, varargin)

gui_StateFields =  {'gui_Name'
    'gui_Singleton'
    'gui_OpeningFcn'
    'gui_OutputFcn'
    'gui_LayoutFcn'
    'gui_Callback'};
gui_Mfile = '';
for i=1:length(gui_StateFields)
    if ~isfield(gui_State, gui_StateFields{i})
        error(message('MATLAB:guide:StateFieldNotFound', gui_StateFields{ i }, gui_Mfile));
    elseif isequal(gui_StateFields{i}, 'gui_Name')
        gui_Mfile = [gui_State.(gui_StateFields{i}), '.m'];
    end
end

numargin = length(varargin);

if numargin == 0
    % NM_OOCV_MANAGER
    % create the GUI only if we are not in the process of loading it
    % already
    gui_Create = true;
elseif local_isInvokeActiveXCallback(gui_State, varargin{:})
    % NM_OOCV_MANAGER(ACTIVEX,...)
    vin{1} = gui_State.gui_Name;
    vin{2} = [get(varargin{1}.Peer, 'Tag'), '_', varargin{end}];
    vin{3} = varargin{1};
    vin{4} = varargin{end-1};
    vin{5} = guidata(varargin{1}.Peer);
    feval(vin{:});
    return;
elseif local_isInvokeHGCallback(gui_State, varargin{:})
    % NM_OOCV_MANAGER('CALLBACK',hObject,eventData,handles,...)
    gui_Create = false;
else
    % NM_OOCV_MANAGER(...)
    % create the GUI and hand varargin to the openingfcn
    gui_Create = true;
end

if ~gui_Create
    % In design time, we need to mark all components possibly created in
    % the coming callback evaluation as non-serializable. This way, they
    % will not be brought into GUIDE and not be saved in the figure file
    % when running/saving the GUI from GUIDE.
    designEval = false;
    if (numargin>1 && ishghandle(varargin{2}))
        fig = varargin{2};
        while ~isempty(fig) && ~ishghandle(fig,'figure')
            fig = get(fig,'parent');
        end
        
        designEval = isappdata(0,'CreatingGUIDEFigure') || (isscalar(fig)&&isprop(fig,'GUIDEFigure'));
    end
        
    if designEval
        beforeChildren = findall(fig);
    end
    
    % evaluate the callback now
    varargin{1} = gui_State.gui_Callback;
    if nargout
        [varargout{1:nargout}] = feval(varargin{:});
    else       
        feval(varargin{:});
    end
    
    % Set serializable of objects created in the above callback to off in
    % design time. Need to check whether figure handle is still valid in
    % case the figure is deleted during the callback dispatching.
    if designEval && ishghandle(fig)
        set(setdiff(findall(fig),beforeChildren), 'Serializable','off');
    end
else
    if gui_State.gui_Singleton
        gui_SingletonOpt = 'reuse';
    else
        gui_SingletonOpt = 'new';
    end

    % Check user passing 'visible' P/V pair first so that its value can be
    % used by oepnfig to prevent flickering
    gui_Visible = 'auto';
    gui_VisibleInput = '';
    for index=1:2:length(varargin)
        if length(varargin) == index || ~ischar(varargin{index})
            break;
        end

        % Recognize 'visible' P/V pair
        len1 = min(length('visible'),length(varargin{index}));
        len2 = min(length('off'),length(varargin{index+1}));
        if ischar(varargin{index+1}) && strncmpi(varargin{index},'visible',len1) && len2 > 1
            if strncmpi(varargin{index+1},'off',len2)
                gui_Visible = 'invisible';
                gui_VisibleInput = 'off';
            elseif strncmpi(varargin{index+1},'on',len2)
                gui_Visible = 'visible';
                gui_VisibleInput = 'on';
            end
        end
    end
    
    % Open fig file with stored settings.  Note: This executes all component
    % specific CreateFunctions with an empty HANDLES structure.

    
    % Do feval on layout code in m-file if it exists
    gui_Exported = ~isempty(gui_State.gui_LayoutFcn);
    % this application data is used to indicate the running mode of a GUIDE
    % GUI to distinguish it from the design mode of the GUI in GUIDE. it is
    % only used by actxproxy at this time.   
    setappdata(0,genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]),1);
    if gui_Exported
        gui_hFigure = feval(gui_State.gui_LayoutFcn, gui_SingletonOpt);

        % make figure invisible here so that the visibility of figure is
        % consistent in OpeningFcn in the exported GUI case
        if isempty(gui_VisibleInput)
            gui_VisibleInput = get(gui_hFigure,'Visible');
        end
        set(gui_hFigure,'Visible','off')

        % openfig (called by local_openfig below) does this for guis without
        % the LayoutFcn. Be sure to do it here so guis show up on screen.
        movegui(gui_hFigure,'onscreen');
    else
        gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt, gui_Visible);
        % If the figure has InGUIInitialization it was not completely created
        % on the last pass.  Delete this handle and try again.
        if isappdata(gui_hFigure, 'InGUIInitialization')
            delete(gui_hFigure);
            gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt, gui_Visible);
        end
    end
    if isappdata(0, genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]))
        rmappdata(0,genvarname(['OpenGuiWhenRunning_', gui_State.gui_Name]));
    end

    % Set flag to indicate starting GUI initialization
    setappdata(gui_hFigure,'InGUIInitialization',1);

    % Fetch GUIDE Application options
    gui_Options = getappdata(gui_hFigure,'GUIDEOptions');
    % Singleton setting in the GUI MATLAB code file takes priority if different
    gui_Options.singleton = gui_State.gui_Singleton;

    if ~isappdata(gui_hFigure,'GUIOnScreen')
        % Adjust background color
        if gui_Options.syscolorfig
            set(gui_hFigure,'Color', get(0,'DefaultUicontrolBackgroundColor'));
        end

        % Generate HANDLES structure and store with GUIDATA. If there is
        % user set GUI data already, keep that also.
        data = guidata(gui_hFigure);
        handles = guihandles(gui_hFigure);
        if ~isempty(handles)
            if isempty(data)
                data = handles;
            else
                names = fieldnames(handles);
                for k=1:length(names)
                    data.(char(names(k)))=handles.(char(names(k)));
                end
            end
        end
        guidata(gui_hFigure, data);
    end

    % Apply input P/V pairs other than 'visible'
    for index=1:2:length(varargin)
        if length(varargin) == index || ~ischar(varargin{index})
            break;
        end

        len1 = min(length('visible'),length(varargin{index}));
        if ~strncmpi(varargin{index},'visible',len1)
            try set(gui_hFigure, varargin{index}, varargin{index+1}), catch break, end
        end
    end

    % If handle visibility is set to 'callback', turn it on until finished
    % with OpeningFcn
    gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
    if strcmp(gui_HandleVisibility, 'callback')
        set(gui_hFigure,'HandleVisibility', 'on');
    end

    feval(gui_State.gui_OpeningFcn, gui_hFigure, [], guidata(gui_hFigure), varargin{:});

    if isscalar(gui_hFigure) && ishghandle(gui_hFigure)
        % Handle the default callbacks of predefined toolbar tools in this
        % GUI, if any
        guidemfile('restoreToolbarToolPredefinedCallback',gui_hFigure); 
        
        % Update handle visibility
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);

        % Call openfig again to pick up the saved visibility or apply the
        % one passed in from the P/V pairs
        if ~gui_Exported
            gui_hFigure = local_openfig(gui_State.gui_Name, 'reuse',gui_Visible);
        elseif ~isempty(gui_VisibleInput)
            set(gui_hFigure,'Visible',gui_VisibleInput);
        end
        if strcmpi(get(gui_hFigure, 'Visible'), 'on')
            figure(gui_hFigure);
            
            if gui_Options.singleton
                setappdata(gui_hFigure,'GUIOnScreen', 1);
            end
        end

        % Done with GUI initialization
        if isappdata(gui_hFigure,'InGUIInitialization')
            rmappdata(gui_hFigure,'InGUIInitialization');
        end

        % If handle visibility is set to 'callback', turn it on until
        % finished with OutputFcn
        gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
        if strcmp(gui_HandleVisibility, 'callback')
            set(gui_hFigure,'HandleVisibility', 'on');
        end
        gui_Handles = guidata(gui_hFigure);
    else
        gui_Handles = [];
    end

    if nargout
        [varargout{1:nargout}] = feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    else
        feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    end

    if isscalar(gui_hFigure) && ishghandle(gui_hFigure)
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
    end
end

function gui_hFigure = local_openfig(name, singleton, visible)

% openfig with three arguments was new from R13. Try to call that first, if
% failed, try the old openfig.
if nargin('openfig') == 2
    % OPENFIG did not accept 3rd input argument until R13,
    % toggle default figure visible to prevent the figure
    % from showing up too soon.
    gui_OldDefaultVisible = get(0,'defaultFigureVisible');
    set(0,'defaultFigureVisible','off');
    gui_hFigure = matlab.hg.internal.openfigLegacy(name, singleton);
    set(0,'defaultFigureVisible',gui_OldDefaultVisible);
else
    % Call version of openfig that accepts 'auto' option"
    gui_hFigure = matlab.hg.internal.openfigLegacy(name, singleton, visible);  
%     %workaround for CreateFcn not called to create ActiveX
%         peers=findobj(findall(allchild(gui_hFigure)),'type','uicontrol','style','text');    
%         for i=1:length(peers)
%             if isappdata(peers(i),'Control')
%                 actxproxy(peers(i));
%             end            
%         end
end

function result = local_isInvokeActiveXCallback(gui_State, varargin)

try
    result = ispc && iscom(varargin{1}) ...
             && isequal(varargin{1},gcbo);
catch
    result = false;
end

function result = local_isInvokeHGCallback(gui_State, varargin)

try
    fhandle = functions(gui_State.gui_Callback);
    result = ~isempty(findstr(gui_State.gui_Name,fhandle.file)) || ...
             (ischar(varargin{1}) ...
             && isequal(ishghandle(varargin{2}), 1) ...
             && (~isempty(strfind(varargin{1},[get(varargin{2}, 'Tag'), '_'])) || ...
                ~isempty(strfind(varargin{1}, '_CreateFcn'))) );
catch
    result = false;
end


