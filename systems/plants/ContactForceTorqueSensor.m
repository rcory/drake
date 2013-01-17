classdef ContactForceTorqueSensor < TimeSteppingRigidBodySensor
  
  properties
    frame;
    body
    normal_ind=[];
    tangent_ind=[];
    jsign=1;
    T % change from body coordinates to coordinates of the sensor
    xyz;
  end
  
  methods
    function obj = ContactForceTorqueSensor(tsmanip,body,xyz,rpy)
      typecheck(body,{'RigidBody','char','numeric'});
      obj.body = body;
      
      if tsmanip.twoD
        if (nargin<3) xyz=zeros(2,1);
        else sizecheck(xyz,[2,1]); end
        if (nargin<4) rpy=0;
        else sizecheck(rpy,1); end
        T = inv([rotmat(rpy),xyz; 0,0,1]);
      else
        if (nargin<3) xyz=zeros(3,1);
        else sizecheck(xyz,[3,1]); end
        if (nargin<4) rpy=zeros(3,1);
        else sizecheck(rpy,[3,1]); end
        T = inv([rotz(rpy(3))*roty(rpy(2))*rotx(rpy(1)),xyz; 0,0,0,1]);
      end      
      
      obj.T = T;
      obj.xyz = xyz;
    end
    
    function tf = isDirectFeedthrough(obj)
      tf = true;
    end
    
    function obj = compile(obj,tsmanip,manip)
      if isa(obj.body,'char')
        obj.body = findLink(tsmanip,obj.body,true);
      elseif isa(body,'numeric')
        obj.body = p.manip.body(body);
      end

      typecheck(obj.body,'RigidBody');
      
      if isempty(obj.body.contact_pts)
        error('Drake:ContactForceTorqueSensor:NoContactPts','There are no contact points associated with body %s',body.name);
      end

      if tsmanip.twoD
        coords{1}=['force_',manip.x_axis_label];
        coords{2}=['force_',manip.y_axis_label];
        coords{3}='torque';
      else
        coords{1}='force_x';
        coords{2}='force_y';
        coords{3}='force_z';
        coords{4}='torque_x';
        coords{5}='torque_y';
        coords{6}='torque_z';
      end
      obj.frame = CoordinateFrame([obj.body.linkname,'ForceTorqueSensor'],length(coords),'f',coords);

      nL = sum([manip.joint_limit_min~=-inf;manip.joint_limit_max~=inf]); % number of joint limits
      nC = manip.num_contacts;
      nP = 2*manip.num_position_constraints;  % number of position constraints
      nV = manip.num_velocity_constraints;  

      body_ind = find(manip.body==obj.body,1);
      num_body_contacts = size(obj.body.contact_pts,2);
      contact_ind_offset = size([manip.body(1:body_ind-1).contact_pts],2);
      
      % z(nL+nP+(1:nC)) = cN
      obj.normal_ind = nL+nP+contact_ind_offset+(1:num_body_contacts);
      
      mC = 2*length(manip.surfaceTangents(manip.gravity)); % get number of tangent vectors

      % z(nL+nP+nC+(1:mC*nC)) = [beta_1;...;beta_mC]
      for i=1:mC
        obj.tangent_ind{i} = nL+nP+(mC*nC)+contact_ind_offset+(1:num_body_contacts);
      end
      
      if isa(manip,'PlanarRigidBodyManipulator')
        obj.jsign=sign(dot(manip.view_axis,[0;-1;0]));
      end
    end
    
    function y = output(obj,tsmanip,manip,t,x,u)
      z = tsmanip.solveLCP(t,x,u)/tsmanip.timestep;
      
      % todo: could do this more efficiently by only computing everything
      % below for indices where the normal forces are non-zero

      % todo: enable mex here (by implementing the mex version of bodyKin)
      use_mex = false;
      kinsol = doKinematics(manip,x(1:manip.getNumDOF),false,use_mex);
      contact_pos = forwardKin(manip,kinsol,obj.body,obj.body.contact_pts);
      
      [d,N] = size(contact_pos);
      [pos,~,normal] = collisionDetect(manip,contact_pos);

      % flip to body coordinates
      pos = sensorKin(obj,manip,kinsol,pos);
      sensor_pos = forwardKin(manip,kinsol,obj.body,obj.xyz);
      normal = sensorKin(obj,manip,kinsol,repmat(sensor_pos,1,N)+normal);
      tangent = manip.surfaceTangents(normal);

      % compute all individual contact forces in sensor coordinates
      force = repmat(z(obj.normal_ind)',d,1).*normal;
      mC=length(tangent);
      for i=1:mC
        force = force + repmat(z(obj.tangent_ind{i})',d,1).*tangent{i} ...
          - repmat(z(obj.tangent_ind{i+mC})',d,1).*tangent{i}; 
      end
      y = sum(force,2);

      if (d==2)
        torque = sum(cross([pos;zeros(1,N)],[force;zeros(1,N)]),2);
        y(3) = obj.jsign*torque(3);
      else
        y(4:6) = sum(cross(pos,force),2);
      end
    end
    
    function fr = getFrame(obj,manip)
      fr = obj.frame;
    end
    
  end
    
  methods (Access=private)
    function pts = sensorKin(obj,manip,kinsol,pts)
      % convert from global frame to sensor frame
      N = size(pts,2);
      pts = obj.T*[bodyKin(manip,kinsol,obj.body,pts);ones(1,N)];
      pts(end,:)=[];
    end
  end
  
end