require 'octokit'

github = Octokit::Client.new( :access_token => "ef581aa5b12118375feb2818f2c7f0ae6eb266cd" ) # Octokit 2.x

repo = 'mc-notes/Issue1'
ref = 'heads/master'

sha_latest_commit = github.ref(repo, ref).object.sha

puts sha_latest_commit