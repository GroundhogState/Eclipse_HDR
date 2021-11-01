% eclipse_init
img_out = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out\img';
data_out = 'C:\Users\jaker\Pictures\Oregon_Eclipse\code\matlab\out\data';
raw_dir = 'C:\Users\jaker\Pictures\Oregon_Eclipse\raw';
dark_raw = 'C:\Users\jaker\Pictures\Oregon_Eclipse\darkfield\raw';

offsets = load(fullfile(data_out,'offsets.mat'));
offsets = offsets.offsets;
etimes = exposure_times();
% Load the hotpix detected by find_hotpix. These are array indices, i.e. in
% the order (Y,X) for images
hotpix = load('out/data/hotpix');
hotpix = hotpix.hotpix_detected;


% Get the two test images 
fnames = get_files(raw_dir,'CR2');
cli_header(2, 'Setup complete');