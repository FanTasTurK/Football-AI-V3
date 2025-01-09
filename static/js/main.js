$(document).ready(function() {
    // Form submit olayını yakala
    $('#prediction-form').on('submit', function(e) {
        e.preventDefault();
        
        // Seçilen takımları al
        const homeTeam = $('#home-team').val();
        const awayTeam = $('#away-team').val();
        
        // Aynı takım seçilmişse uyarı ver
        if (homeTeam === awayTeam) {
            alert('Lütfen farklı takımlar seçin!');
            return;
        }
        
        // Tahmin butonunu devre dışı bırak
        $('button[type="submit"]').prop('disabled', true).html(
            '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Tahmin yapılıyor...'
        );
        
        // API'ye istek at
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                home_team: homeTeam,
                away_team: awayTeam
            }),
            success: function(response) {
                // Sonuçları göster
                $('#home-win-prob').text(response.match_result.home_win);
                $('#draw-prob').text(response.match_result.draw);
                $('#away-win-prob').text(response.match_result.away_win);
                
                $('#score-pred').text(
                    `${response.score.prediction} (${response.score.probability})`
                );
                $('#htft-pred').text(
                    `${response.htft.prediction} (${response.htft.probability})`
                );
                $('#btts-pred').text(
                    `${response.btts.prediction} (${response.btts.probability})`
                );
                
                // Sonuç kartını göster
                $('#results').slideDown();
                
                // Tahmin butonunu tekrar aktif et
                $('button[type="submit"]').prop('disabled', false).text('Tahmin Yap');
            },
            error: function(xhr, status, error) {
                // Hata mesajını göster
                alert('Tahmin yapılırken bir hata oluştu: ' + error);
                
                // Tahmin butonunu tekrar aktif et
                $('button[type="submit"]').prop('disabled', false).text('Tahmin Yap');
            }
        });
    });
    
    // Takım seçimi değiştiğinde diğer takımı otomatik olarak kaldır
    $('#home-team').change(function() {
        const selectedTeam = $(this).val();
        const awayTeam = $('#away-team');
        
        // Eğer seçilen takım deplasman takımı olarak seçilmişse, onu kaldır
        if (awayTeam.val() === selectedTeam) {
            awayTeam.val('');
        }
        
        // Seçilen takımı deplasman listesinde devre dışı bırak
        awayTeam.find('option').prop('disabled', false);
        if (selectedTeam) {
            awayTeam.find(`option[value="${selectedTeam}"]`).prop('disabled', true);
        }
    });
    
    $('#away-team').change(function() {
        const selectedTeam = $(this).val();
        const homeTeam = $('#home-team');
        
        // Eğer seçilen takım ev sahibi takım olarak seçilmişse, onu kaldır
        if (homeTeam.val() === selectedTeam) {
            homeTeam.val('');
        }
        
        // Seçilen takımı ev sahibi listesinde devre dışı bırak
        homeTeam.find('option').prop('disabled', false);
        if (selectedTeam) {
            homeTeam.find(`option[value="${selectedTeam}"]`).prop('disabled', true);
        }
    });
}); 