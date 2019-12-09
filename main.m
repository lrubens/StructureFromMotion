clear
close all
imtool close all

DEBUG = 0;
image_dir = "hotel/";
images = get_images(image_dir);

structure_from_motion = SFM(images, image_dir, DEBUG);
structure_from_motion = structure_from_motion.init_tracking();
structure_from_motion = structure_from_motion.klt_tracker();
[M, S] = structure_from_motion.factorization();

S = S';
ptCloud = pointCloud(S);
pcshow(ptCloud);
pcwrite(ptCloud, 'results/output.ply', 'PLYFormat', 'ascii');

%% Utility Functions
function images = get_images(dir)
  if ~isfolder(dir)
    sprintf("Directory does not exist\n");
    return;
  end
  images = imageDatastore(dir);
end
