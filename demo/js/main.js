$(function () {
	'use strict';

	var window_width = $(window).width(),
		window_height = window.innerHeight,
		header_height = $('.default-header').height(),
		header_height_static = $('.site-header.static').outerHeight(),
		fitscreen = window_height - header_height;

	$('.fullscreen').css('height', window_height);
	$('.fitscreen').css('height', fitscreen);

	$('#gt-clouds').on('click', function () {
		alert("This algorithm is demoed on MATLAB.")
	});

	$("#gt-lrcn").on('click', function () {
		setTimeout(function () {
			$('html,body').animate({
				scrollTop: $("#demo-area").offset().top
			}, 'slow');
		}, 0);
	});

	$('#in-video').on('change', function () {
		if (this.files.length <= 0)
			return;
		var inVideo = this.files[0];
		var videoUrl = URL.createObjectURL(inVideo);
		setTimeout(function () {
			$('#in-video-src').attr('src', 'https://www.youtube.com/embed/F9Bo89m2f6g');
		}, 0);
	})

});
