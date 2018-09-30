
<?php
 	$imgPath = $_POST['imagePath'];
 	$imgName = $_POST['imageName'];
 	//$imgName = escapeshellarg($imgName);
 	//$imgPath = escapeshellarg($imgPath);

 	echo $imgName;
 	echo $imgPath;
  //int id = rand(0,10000);
   $output=shell_exec("./script/resizer2.sh '$imgPath' '$imgName'");
   echo ('byyyye');
   echo $output;
//Obviously, this is a temporary location :D

// 168.62.177.216 -->