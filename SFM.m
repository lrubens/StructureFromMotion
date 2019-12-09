classdef SFM
  %frames Summary of this class goes here
  %   Detailed explanation goes here

  properties
    video_out
    frames
    num_frames
    frame_rate
    features
    tracked_X
    tracked_Y
    W
    M
    S
    dir
    DEBUG
  end

  methods
    function obj = SFM(images, dir, DEBUG)
      %Frames Construct an instance of this class
      if nargin < 3
        DEBUG = 0;
      end
      obj.DEBUG = DEBUG;
      obj.frames = images;
      obj.dir = dir;
      obj.num_frames = length(obj.frames.Files);
      obj.video_out = 'video/hotel.avi';
    end

    function obj = init_tracking(obj)
      hotel_video = VideoWriter(obj.video_out);
      open(hotel_video);
      for i = 1:obj.num_frames
        writeVideo(hotel_video, readimage(obj.frames, i));
      end
      close(hotel_video);
    end

    function obj = klt_tracker(obj)
      corners_threshold = 1000;
      video_reader = vision.VideoFileReader(obj.video_out);
      video_frame = video_reader();
      points = detectHarrisFeatures(rgb2gray(video_frame));
      points = points.selectStrongest(corners_threshold).Location;
      tracker = vision.PointTracker('MaxBidirectionalError', 1);
      initialize(tracker, points, video_frame);
      removed_index = [];
      found_points = [];
      found_points = [found_points; points];
      
      % Performing KLT Tracking
      while ~isDone(video_reader)
        video_frame = video_reader();
        [points, validity] = tracker(video_frame);
        removed_index = unique([removed_index; find(validity == 0)]);
        if obj.DEBUG == 1
          visible_points = points(validity, :);
          im = insertMarker(video_frame, visible_points, '+', 'Color', 'white');
          imshow(im);
        end
        found_points = cat(3, found_points, points);
      end
      for i = 1:obj.num_frames
        [temp_points, ~] = removerows(found_points(:, :, i), 'ind', removed_index);
        obj.tracked_X = vertcat(obj.tracked_X, temp_points(:, 1)');
        obj.tracked_Y = vertcat(obj.tracked_Y, temp_points(:, 2)');
      end
      
      % Eliminating Translation by subtracting the mean of each row
      obj.tracked_X = bsxfun(@minus, obj.tracked_X, mean(obj.tracked_X, 2));
      obj.tracked_Y = bsxfun(@minus, obj.tracked_Y, mean(obj.tracked_Y, 2));
      
      % Creating W
      obj.W = [obj.tracked_X; obj.tracked_Y];
      release(video_reader);
      release(tracker);
    end
    
    function [M, S] = factorization(obj)
      F = obj.num_frames;
      [U, D, V] = svd(obj.W);
      M_ = U(:, 1:3)*(D(1:3, 1:3) ^ 1/2); 
      S_ = (D(1:3, 1:3) ^ 1/2)*V(:, 1:3)';
      
      Is = M_(1:F, :);
      Js = M_(F+1:end, :);

      gfun = @(a, b)[ a(1)*b(1), a(1)*b(2)+a(2)*b(1), a(1)*b(3)+a(3)*b(1), ...
                    a(2)*b(2), a(2)*b(3)+a(3)*b(2), a(3)*b(3)];
      G = zeros(3*F, 6);
      for f = 1:3*F
          if f <= F
              G(f, :) = gfun(Is(f,:), Is(f,:));
          elseif f <= 2*F
              G(f, :) = gfun(Js(mod(f, F+1)+1, :), Js(mod(f, F+1)+1, :));
          else
              G(f, :) = gfun(Is(mod(f, 2*F),:), Js(mod(f, 2*F),:));
          end
      end

      c = [ones(2*F, 1); zeros(F, 1)];

      % Use Cholesky to obtain Q
      [U, S, V] = svd(G);
      l_ = U'*c;
      y = [l_(1)/S(1,1); l_(2)/S(2,2); l_(3)/S(3,3); l_(4)/S(4,4); ...
          l_(5)/S(5,5); l_(6)/S(6,6)];
      l = V*y;
      L = [l(1) l(2) l(3);
           l(2) l(4) l(5);
           l(3) l(5) l(6)];
      Q = chol(L); 

      % Compute M and S
      M = M_*Q;
      S = inv(Q)*S_;
      
      if obj.DEBUG == 1
        % plot of 3D points
        figure;
        plot3(S_(1, :), S_(2,:), S_(3,:),'k.'); hold on;
        plot3(S(1, :), S(2,:), S(3,:),'b.');
        plot3(0,0,0,'gs');
        grid on; 
      end
    end

  end
end
